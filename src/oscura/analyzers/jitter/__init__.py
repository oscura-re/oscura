"""Jitter analysis module for advanced timing characterization.

This module provides IEEE 2414-2020 compliant jitter analysis including
decomposition into random and deterministic components, bathtub curves,
and jitter spectrum analysis.


Example:
    >>> from oscura.analyzers.jitter import extract_rj, tj_at_ber, bathtub_curve
    >>> rj = extract_rj(tie_data)
    >>> tj = tj_at_ber(rj_rms=rj.rj_rms, dj_pp=dj.dj_pp, ber=1e-12)
    >>> positions, ber_values = bathtub_curve(tie_data, unit_interval=1e-9)

References:
    IEEE 2414-2020: Standard for Jitter and Phase Noise
    JEDEC JESD65C: Definition of Skew Specifications for Standard Logic Devices
"""

from oscura.analyzers.jitter.ber import (
    BathtubCurveResult,
    bathtub_curve,
    ber_from_q_factor,
    eye_opening_at_ber,
    q_factor_from_ber,
    tj_at_ber,
)
from oscura.analyzers.jitter.classification import (
    JitterClassificationResult,
    JitterComponentEstimate,
)
from oscura.analyzers.jitter.decomposition import (
    DataDependentJitterResult,
    DeterministicJitterResult,
    JitterDecomposition,
    PeriodicJitterResult,
    RandomJitterResult,
    decompose_jitter,
    extract_ddj,
    extract_dj,
    extract_pj,
    extract_rj,
)
from oscura.analyzers.jitter.measurements import (
    CycleJitterResult,
    DutyCycleDistortionResult,
    cycle_to_cycle_jitter,
    measure_dcd,
    period_jitter,
    tie_from_edges,
)
from oscura.analyzers.jitter.spectrum import (
    JitterSpectrumResult,
    identify_periodic_components,
    jitter_spectrum,
)

__all__ = [
    "BathtubCurveResult",
    "CycleJitterResult",
    "DataDependentJitterResult",
    "DeterministicJitterResult",
    "DutyCycleDistortionResult",
    "JitterClassificationResult",
    "JitterComponentEstimate",
    "JitterDecomposition",
    "JitterSpectrumResult",
    "PeriodicJitterResult",
    "RandomJitterResult",
    "bathtub_curve",
    "ber_from_q_factor",
    "cycle_to_cycle_jitter",
    "decompose_jitter",
    "extract_ddj",
    "extract_dj",
    "extract_pj",
    "extract_rj",
    "eye_opening_at_ber",
    "identify_periodic_components",
    "jitter_spectrum",
    "measure_dcd",
    "period_jitter",
    "q_factor_from_ber",
    "tie_from_edges",
    "tj_at_ber",
]
