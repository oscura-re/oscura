"""Reverse Engineering Demonstrations.

This package contains demonstrations of Oscura's reverse engineering capabilities:

1. 01_unknown_protocol.py - Complete unknown protocol reverse engineering workflow
2. 02_crc_recovery.py - CRC polynomial recovery from message samples
3. 03_state_machines.py - State machine learning using L* algorithm
4. 04_field_inference.py - Automatic field boundary and type detection
5. 05_pattern_discovery.py - Message pattern recognition and analysis
6. 06_wireshark_export.py - Wireshark dissector generation from inferred protocols

All demonstrations follow the BaseDemo template and include:
- Synthetic data generation
- Complete workflow execution
- Result validation
- Clear output formatting
"""

__all__ = [
    "CRCRecoveryDemo",
    "FieldInferenceDemo",
    "PatternDiscoveryDemo",
    "StateMachineLearningDemo",
    "UnknownProtocolDemo",
    "WiresharkExportDemo",
]
