"""Filter convenience functions namespace.

This module provides a namespace for filter functions to support:
    from oscura.filtering import filters
    filters.low_pass(trace, cutoff=1000)

Re-exports convenience functions from the filtering package.
"""

from oscura.filtering.convenience import (
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

__all__ = [
    "band_pass",
    "band_stop",
    "high_pass",
    "low_pass",
    "matched_filter",
    "median_filter",
    "moving_average",
    "notch_filter",
    "savgol_filter",
]
