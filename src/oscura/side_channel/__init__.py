"""Side-channel analysis for cryptographic implementation attacks.

.. deprecated:: 0.6.0
    This module is deprecated. Use :mod:`oscura.analyzers.side_channel` instead.
    This module will be removed in v1.0.0.

This package provides tools for performing side-channel attacks on cryptographic
implementations using power analysis, electromagnetic analysis, timing analysis,
and fault injection techniques.

Migration Guide:
    Old import (deprecated):
        >>> from oscura.side_channel.dpa import DPAAnalyzer, PowerTrace

    New import (recommended):
        >>> from oscura.analyzers.side_channel import DPAAnalyzer, CPAAnalyzer
        >>> from oscura.analyzers.side_channel.power import hamming_weight

Note:
    The oscura.side_channel.dpa module contains a different DPAAnalyzer implementation
    than oscura.analyzers.side_channel.power. The older implementation in this module
    (oscura.side_channel.dpa.DPAAnalyzer) supports combined DPA/CPA/Template attacks
    with a single class. The newer implementation in oscura.analyzers.side_channel
    provides separate DPAAnalyzer and CPAAnalyzer classes with cleaner APIs.

    For new code, use the oscura.analyzers.side_channel module.
    For existing code using DPAAnalyzer with attack_type parameter, continue using
    oscura.side_channel.dpa until migration to the new API.

Example (new API - recommended):
    >>> from oscura.analyzers.side_channel import DPAAnalyzer, CPAAnalyzer
    >>> # DPA attack
    >>> dpa = DPAAnalyzer(target_bit=0)
    >>> result = dpa.analyze(traces, plaintexts)
    >>> # CPA attack
    >>> cpa = CPAAnalyzer(leakage_model="hamming_weight")
    >>> result = cpa.analyze(traces, plaintexts)

Example (old API - deprecated):
    >>> from oscura.side_channel.dpa import DPAAnalyzer, PowerTrace
    >>> analyzer = DPAAnalyzer(attack_type="cpa", leakage_model="hamming_weight")
    >>> traces = [PowerTrace(timestamp=t, power=p, plaintext=pt) for ...]
    >>> result = analyzer.perform_attack(traces, target_byte=0)
"""

from __future__ import annotations

import warnings

# Issue deprecation warning on import
warnings.warn(
    "oscura.side_channel is deprecated and will be removed in v1.0.0. "
    "Use oscura.analyzers.side_channel instead. "
    "See migration guide in module docstring.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from the local dpa module for backward compatibility
# The dpa.py module in this directory contains the legacy implementation
from oscura.side_channel.dpa import DPAAnalyzer, DPAResult, PowerTrace

__all__ = ["DPAAnalyzer", "DPAResult", "PowerTrace"]
