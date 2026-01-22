"""Common utilities for Oscura demonstrations."""

from demonstrations.common.base_demo import BaseDemo
from demonstrations.common.data_generation import (
    add_noise,
    generate_complex_signal,
    generate_pulse_train,
    generate_sine_wave,
    generate_square_wave,
)
from demonstrations.common.formatting import (
    format_duration,
    format_list,
    format_percentage,
    format_size,
    format_table,
    format_value,
)
from demonstrations.common.plotting import (
    plot_comparison,
    plot_spectrum,
    plot_waveform,
)
from demonstrations.common.validation import (
    validate_approximately,
    validate_exists,
    validate_length,
    validate_range,
    validate_results,
    validate_type,
)

__all__ = [
    # Base class
    "BaseDemo",
    # Data generation
    "generate_sine_wave",
    "generate_square_wave",
    "generate_pulse_train",
    "generate_complex_signal",
    "add_noise",
    # Formatting
    "format_value",
    "format_percentage",
    "format_table",
    "format_duration",
    "format_size",
    "format_list",
    # Plotting
    "plot_waveform",
    "plot_spectrum",
    "plot_comparison",
    # Validation
    "validate_results",
    "validate_range",
    "validate_exists",
    "validate_length",
    "validate_type",
    "validate_approximately",
]
