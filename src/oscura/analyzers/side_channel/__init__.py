"""Side-channel analysis module.

This module provides research-grade implementations of side-channel analysis
techniques including Differential Power Analysis (DPA), Correlation Power
Analysis (CPA), and timing analysis.

Example:
    >>> from oscura.analyzers.side_channel import DPAAnalyzer, CPAAnalyzer
    >>> # DPA attack
    >>> dpa = DPAAnalyzer(target_bit=0)
    >>> result = dpa.analyze(traces, plaintexts)
    >>> print(f"Key byte guess: 0x{result.key_guess:02X}")
    >>>
    >>> # CPA attack
    >>> cpa = CPAAnalyzer(leakage_model="hamming_weight")
    >>> result = cpa.analyze(traces, plaintexts)
    >>> print(f"Correlation: {result.max_correlation:.4f}")

References:
    Kocher et al. "Differential Power Analysis" (CRYPTO 1999)
    Brier et al. "Correlation Power Analysis" (CHES 2004)
"""

from oscura.analyzers.side_channel.power import (
    CPAAnalyzer,
    CPAResult,
    DPAAnalyzer,
    DPAResult,
    LeakageModel,
    hamming_distance,
    hamming_weight,
)
from oscura.analyzers.side_channel.timing import (
    TimingAnalyzer,
    TimingAttackResult,
    TimingLeak,
)

__all__ = [
    # Power analysis
    "CPAAnalyzer",
    "CPAResult",
    "DPAAnalyzer",
    "DPAResult",
    "LeakageModel",
    # Timing analysis
    "TimingAnalyzer",
    "TimingAttackResult",
    "TimingLeak",
    "hamming_distance",
    "hamming_weight",
]
