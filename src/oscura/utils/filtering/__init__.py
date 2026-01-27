"""Signal filtering module for Oscura.

Provides digital filter design, application, and introspection capabilities
including IIR and FIR filters, various filter types (Butterworth, Chebyshev,
Bessel, Elliptic), and convenience filters (moving average, median).


Example:
    >>> from oscura.utils.filtering import LowPassFilter, design_filter
    >>> lpf = LowPassFilter(cutoff=1e6, sample_rate=10e6, order=4)
    >>> filtered_trace = lpf.apply(trace)
    >>> w, h = lpf.get_frequency_response()
"""

# Import filters module as namespace for DSL compatibility
from oscura.utils.filtering import filters
from oscura.utils.filtering.base import (
    Filter,
    FIRFilter,
    IIRFilter,
)
from oscura.utils.filtering.convenience import (
    band_pass,
    band_stop,
    high_pass,
    low_pass,
    matched_filter,
    median_filter,
    moving_average,
    notch_filter,
    savgol_filter,
)
from oscura.utils.filtering.design import (
    BandPassFilter,
    BandStopFilter,
    BesselFilter,
    ButterworthFilter,
    ChebyshevType1Filter,
    ChebyshevType2Filter,
    EllipticFilter,
    HighPassFilter,
    LowPassFilter,
    design_filter,
    design_filter_spec,
)
from oscura.utils.filtering.introspection import (
    FilterIntrospection,
    plot_bode,
    plot_impulse,
    plot_poles_zeros,
    plot_step,
)

__all__ = [
    "BandPassFilter",
    "BandStopFilter",
    "BesselFilter",
    "ButterworthFilter",
    "ChebyshevType1Filter",
    "ChebyshevType2Filter",
    "EllipticFilter",
    "FIRFilter",
    # Base classes
    "Filter",
    # Introspection
    "FilterIntrospection",
    "HighPassFilter",
    "IIRFilter",
    # Filter types
    "LowPassFilter",
    "band_pass",
    "band_stop",
    # Design functions
    "design_filter",
    "design_filter_spec",
    "filters",
    "high_pass",
    "low_pass",
    "matched_filter",
    "median_filter",
    # Convenience functions
    "moving_average",
    "notch_filter",
    "plot_bode",
    "plot_impulse",
    "plot_poles_zeros",
    "plot_step",
    "savgol_filter",
]
