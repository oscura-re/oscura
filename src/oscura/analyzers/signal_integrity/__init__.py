"""Signal integrity analysis module.

This module provides S-parameter handling, channel de-embedding,
and equalization functions for high-speed serial link analysis.


Example:
    >>> from oscura.analyzers.signal_integrity import (
    ...     load_touchstone, deembed, ffe_equalize, return_loss
    ... )
    >>> s_params = load_touchstone("cable.s2p")
    >>> clean_trace = deembed(trace, s_params)
    >>> rl = return_loss(s_params, frequency=1e9)

References:
    Touchstone 2.0 File Format Specification
    IEEE 370-2020: Standard for Electrical Characterization of PCBs
"""

from oscura.analyzers.signal_integrity.embedding import (
    cascade_deembed,
    deembed,
    embed,
)
from oscura.analyzers.signal_integrity.equalization import (
    CTLEResult,
    DFEResult,
    FFEResult,
    ctle_equalize,
    dfe_equalize,
    ffe_equalize,
    optimize_ffe,
)
from oscura.analyzers.signal_integrity.sparams import (
    SParameterData,
    abcd_to_s,
    insertion_loss,
    load_touchstone,
    return_loss,
    s_to_abcd,
)

__all__ = [
    "CTLEResult",
    "DFEResult",
    # Equalization
    "FFEResult",
    # S-parameters
    "SParameterData",
    "abcd_to_s",
    "cascade_deembed",
    "ctle_equalize",
    # Embedding
    "deembed",
    "dfe_equalize",
    "embed",
    "ffe_equalize",
    "insertion_loss",
    "load_touchstone",
    "optimize_ffe",
    "return_loss",
    "s_to_abcd",
]
