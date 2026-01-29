"""Common utilities for Oscura demonstrations."""

from demonstrations.common.base_demo import BaseDemo
from demonstrations.common.builders import SignalBuilder
from demonstrations.common.data_generation import (
    add_noise,
    generate_complex_signal,
    generate_pulse_train,
    generate_sine_wave,
    generate_square_wave,
)
from demonstrations.common.formatting import (
    BLUE,
    GREEN,
    RED,
    RESET,
    YELLOW,
    format_duration,
    format_list,
    format_percentage,
    format_size,
    format_table,
    format_value,
    print_subheader,
)
from demonstrations.common.output import ValidationSuite, print_info, print_result
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
    "BLUE",
    "GREEN",
    "RED",
    "RESET",
    "YELLOW",
    "BaseDemo",
    "SignalBuilder",
    "ValidationSuite",
    "add_noise",
    "format_duration",
    "format_list",
    "format_percentage",
    "format_size",
    "format_table",
    "format_value",
    "generate_complex_signal",
    "generate_pulse_train",
    "generate_sine_wave",
    "generate_square_wave",
    "plot_comparison",
    "plot_spectrum",
    "plot_waveform",
    "print_info",
    "print_result",
    "print_subheader",
    "validate_approximately",
    "validate_exists",
    "validate_length",
    "validate_range",
    "validate_results",
    "validate_type",
]
