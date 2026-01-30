"""Common utilities for Oscura demos.

This module provides shared infrastructure for all demos including:
- BaseDemo abstract class for consistent demo structure
- Console output formatting utilities
- Validation framework
- Plotting helpers (when matplotlib available)
"""

from demos.common.base_demo import BaseDemo, DemoResult, run_demo_main
from demos.common.formatting import (
    print_error,
    print_header,
    print_info,
    print_metric,
    print_result,
    print_subheader,
    print_success,
    print_table,
    print_warning,
)
from demos.common.validation import ValidationResult, ValidationSuite

# Import plotting utilities if matplotlib is available
try:
    from demos.common.plotting import DemoPlotter, create_demo_plotter

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    DemoPlotter = None  # type: ignore[misc, assignment]
    create_demo_plotter = None  # type: ignore[assignment]

__all__ = [
    "HAS_PLOTTING",
    "BaseDemo",
    "DemoPlotter",
    "DemoResult",
    "ValidationResult",
    "ValidationSuite",
    "create_demo_plotter",
    "print_error",
    "print_header",
    "print_info",
    "print_metric",
    "print_result",
    "print_subheader",
    "print_success",
    "print_table",
    "print_warning",
    "run_demo_main",
]
