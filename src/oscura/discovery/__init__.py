"""Auto-discovery and signal characterization.

This module provides intelligent auto-discovery features for non-expert
users, including signal characterization, anomaly detection, quality
assessment, and automatic protocol decoding.


Example:
    >>> import oscura as osc
    >>> trace = osc.load("capture.wfm")
    >>> result = osc.discovery.characterize_signal(trace)
    >>> print(f"Signal type: {result.signal_type} (confidence: {result.confidence:.2f})")

References:
    Oscura Auto-Discovery Requirements
"""

from oscura.discovery.anomaly_detector import (
    Anomaly,
    find_anomalies,
)
from oscura.discovery.auto_decoder import (
    DecodeResult,
    decode_protocol,
)
from oscura.discovery.comparison import (
    Difference,
    TraceDiff,
    compare_traces,
)
from oscura.discovery.quality_validator import (
    DataQuality,
    assess_data_quality,
)
from oscura.discovery.signal_detector import (
    SignalCharacterization,
    characterize_signal,
)

__all__ = [
    "Anomaly",
    "DataQuality",
    "DecodeResult",
    "Difference",
    "SignalCharacterization",
    "TraceDiff",
    "assess_data_quality",
    "characterize_signal",
    "compare_traces",
    "decode_protocol",
    "find_anomalies",
]
