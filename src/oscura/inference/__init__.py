"""Auto-inference and smart defaults for Oscura.

This module provides automatic parameter detection and intelligent defaults
for common analysis tasks.

- RE-SEQ-002: Sequence Pattern Detection
- RE-SEQ-003: Request-Response Correlation
- RE-STR-001: UDP Stream Reconstruction
- RE-STR-002: TCP Stream Reassembly
- RE-STR-003: Message Framing and Segmentation
- RE-BIN-001: Magic Byte Detection
- RE-BIN-002: Structure Alignment Detection
- RE-BIN-003: Binary Parser DSL
- RE-DSL-003: Protocol Format Library
"""

# Active Learning: L* algorithm for DFA inference
from oscura.inference.active_learning import (
    LStarLearner,
    ObservationTable,
    Oracle,
    SimulatorTeacher,
)
from oscura.inference.adaptive_tuning import (
    AdaptiveParameterTuner,
    TunedParameters,
    get_adaptive_parameters,
)
from oscura.inference.alignment import (
    AlignmentResult,
    align_global,
    align_local,
    align_multiple,
    compute_similarity,
    find_conserved_regions,
    find_variable_regions,
)
from oscura.inference.bayesian import (
    BayesianInference,
    Posterior,
    Prior,
    SequentialBayesian,
    infer_with_uncertainty,
)

# RE-BIN-001, RE-BIN-002, RE-BIN-003: Binary Format Inference
from oscura.inference.binary import (
    KNOWN_MAGIC_BYTES,
    AlignmentDetector,
    BinaryParserGenerator,
    MagicByteDetector,
    MagicByteResult,
    ParserDefinition,
    ParserField,
    detect_alignment,
    detect_magic_bytes,
    find_all_magic_bytes,
    generate_parser,
    parser_to_python,
    parser_to_yaml,
)
from oscura.inference.binary import (
    AlignmentResult as BinaryAlignmentResult,
)
from oscura.inference.crc_reverse import (
    STANDARD_CRCS,
    CRCParameters,
    CRCReverser,
    verify_crc,
)
from oscura.inference.logic import detect_logic_family
from oscura.inference.message_format import (
    InferredField,
    MessageFormatInferrer,
    MessageSchema,
    detect_field_types,
    find_dependencies,
    infer_format,
)
from oscura.inference.protocol import detect_protocol
from oscura.inference.protocol_dsl import (
    DecodedMessage,
    FieldDefinition,
    ProtocolDecoder,
    ProtocolDefinition,
    ProtocolEncoder,
    decode_message,
    load_protocol,
)

# RE-DSL-003: Protocol Format Library
from oscura.inference.protocol_library import (
    ProtocolInfo,
    ProtocolLibrary,
    get_decoder,
    get_library,
    get_protocol,
    list_protocols,
)

# RE-SEQ-002, RE-SEQ-003: Sequence Pattern Detection and Correlation
from oscura.inference.sequences import (
    CommunicationFlow,
    RequestResponseCorrelator,
    RequestResponsePair,
    SequencePattern,
    SequencePatternDetector,
    calculate_latency_stats,
    correlate_requests,
    detect_sequence_patterns,
    find_message_dependencies,
)
from oscura.inference.signal_intelligence import (
    AnalysisRecommendation,
    assess_signal_quality,
    check_measurement_suitability,
    classify_signal,
    get_optimal_domain_order,
    recommend_analyses,
    suggest_measurements,
)
from oscura.inference.spectral import auto_spectral_config
from oscura.inference.state_machine import (
    FiniteAutomaton,
    State,
    StateMachine,
    StateMachineExtractor,
    StateMachineInferrer,
    Transition,
    infer_rpni,
    minimize_dfa,
    to_dot,
    to_networkx,
)

# RE-STR-001, RE-STR-002, RE-STR-003: Stream Reassembly
from oscura.inference.stream import (
    FramingResult,
    MessageFrame,
    MessageFramer,
    ReassembledStream,
    StreamSegment,
    TCPStreamReassembler,
    UDPStreamReassembler,
    detect_message_framing,
    extract_messages,
    reassemble_tcp_stream,
    reassemble_udp_stream,
)

__all__ = [
    "KNOWN_MAGIC_BYTES",
    "STANDARD_CRCS",
    "AdaptiveParameterTuner",
    # RE-BIN-001, RE-BIN-002, RE-BIN-003: Binary Format Inference
    "AlignmentDetector",
    "AlignmentResult",
    "AnalysisRecommendation",
    "BayesianInference",
    "BinaryAlignmentResult",
    "BinaryParserGenerator",
    # CRC Reverse Engineering
    "CRCParameters",
    "CRCReverser",
    # RE-SEQ-002, RE-SEQ-003: Sequence Patterns and Correlation
    "CommunicationFlow",
    "DecodedMessage",
    "FieldDefinition",
    "FiniteAutomaton",
    # RE-STR-001, RE-STR-002, RE-STR-003: Stream Reassembly
    "FramingResult",
    "InferredField",
    # Active Learning
    "LStarLearner",
    "MagicByteDetector",
    "MagicByteResult",
    "MessageFormatInferrer",
    "MessageFrame",
    "MessageFramer",
    "MessageSchema",
    "ObservationTable",
    "Oracle",
    "ParserDefinition",
    "ParserField",
    "Posterior",
    "Prior",
    "ProtocolDecoder",
    "ProtocolDefinition",
    "ProtocolEncoder",
    # RE-DSL-003: Protocol Format Library
    "ProtocolInfo",
    "ProtocolLibrary",
    "ReassembledStream",
    "RequestResponseCorrelator",
    "RequestResponsePair",
    "SequencePattern",
    "SequencePatternDetector",
    "SequentialBayesian",
    "SimulatorTeacher",
    "State",
    "StateMachine",
    "StateMachineExtractor",
    "StateMachineInferrer",
    "StreamSegment",
    "TCPStreamReassembler",
    "Transition",
    "TunedParameters",
    "UDPStreamReassembler",
    "align_global",
    "align_local",
    "align_multiple",
    # Original exports
    "assess_signal_quality",
    "auto_spectral_config",
    "calculate_latency_stats",
    "check_measurement_suitability",
    "classify_signal",
    "compute_similarity",
    "correlate_requests",
    "decode_message",
    "detect_alignment",
    "detect_field_types",
    "detect_logic_family",
    "detect_magic_bytes",
    "detect_message_framing",
    "detect_protocol",
    "detect_sequence_patterns",
    "extract_messages",
    "find_all_magic_bytes",
    "find_conserved_regions",
    "find_dependencies",
    "find_message_dependencies",
    "find_variable_regions",
    "generate_parser",
    "get_adaptive_parameters",
    "get_decoder",
    "get_library",
    "get_optimal_domain_order",
    "get_protocol",
    "infer_format",
    "infer_rpni",
    "infer_with_uncertainty",
    "list_protocols",
    "load_protocol",
    "minimize_dfa",
    "parser_to_python",
    "parser_to_yaml",
    "reassemble_tcp_stream",
    "reassemble_udp_stream",
    "recommend_analyses",
    "suggest_measurements",
    "to_dot",
    "to_networkx",
    "verify_crc",
]
