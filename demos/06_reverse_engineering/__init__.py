"""Reverse Engineering Demonstrations.

This package contains demonstrations of protocol reverse engineering,
field inference, CRC recovery, state machine learning, and advanced
inference techniques.

Demonstrations:
- 01_unknown_protocol.py: Complete unknown protocol reverse engineering workflow
- 02_crc_recovery.py: CRC polynomial recovery from message samples
- 03_state_machines.py: Protocol state machine learning from traces
- 04_field_inference.py: Automatic field boundary and type detection
- 05_pattern_discovery.py: Message pattern recognition and analysis
- 06_wireshark_export.py: Wireshark dissector generation
- 07_entropy_analysis.py: Entropy-based protocol analysis
- 08_data_classification.py: Data type classification
- 09_signal_classification.py: ML-based signal classification
- 10_anomaly_detection.py: Anomaly detection in protocol traces
- 11_bayesian_inference.py: Bayesian inference for signal analysis
- 12_active_learning.py: Active learning (L*) for DFA inference
- 13_protocol_dsl.py: Protocol DSL for declarative protocol definition
- 14_comprehensive_re.py: Comprehensive reverse engineering workflow
- 15_re_tool.py: Interactive reverse engineering tool
"""

__all__ = [
    "CRCRecoveryDemo",
    "FieldInferenceDemo",
    "PatternDiscoveryDemo",
    "StateMachineLearningDemo",
    "UnknownProtocolDemo",
]
