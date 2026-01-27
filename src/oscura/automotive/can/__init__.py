"""CAN bus analysis and reverse engineering.

This submodule provides CAN-specific analysis tools for reverse engineering
automotive protocols from captured CAN bus data.
"""

__all__ = [
    "ByteChange",
    "CANMessage",
    "CANMessageList",
    "CANSession",
    "CANStateMachine",
    "DBCGenerator",
    "DBCMessage",
    "DBCNode",
    "DBCSignal",
    "DecodedSignal",
    "FrequencyChange",
    "MessageAnalysis",
    "MessagePair",
    "MessageSequence",
    "PatternAnalyzer",
    "SequenceExtraction",
    "SignalDefinition",
    "StimulusResponseAnalyzer",
    "StimulusResponseReport",
    "TemporalCorrelation",
]

try:
    from oscura.automotive.can.dbc_generator import (
        DBCGenerator,
        DBCMessage,
        DBCNode,
        DBCSignal,
    )
    from oscura.automotive.can.models import (
        CANMessage,
        CANMessageList,
        DecodedSignal,
        MessageAnalysis,
        SignalDefinition,
    )
    from oscura.automotive.can.patterns import (
        MessagePair,
        MessageSequence,
        PatternAnalyzer,
        TemporalCorrelation,
    )
    from oscura.automotive.can.session import CANSession
    from oscura.automotive.can.state_machine import CANStateMachine, SequenceExtraction
    from oscura.automotive.can.stimulus_response import (
        ByteChange,
        FrequencyChange,
        StimulusResponseAnalyzer,
        StimulusResponseReport,
    )
except ImportError:
    # Optional dependencies not installed
    pass
