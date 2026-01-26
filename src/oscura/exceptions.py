"""Deprecated exception module - use oscura.core.exceptions instead.

This module is deprecated and maintained only for backward compatibility.
All exceptions have been moved to oscura.core.exceptions.

Warning:
    This module will be removed in a future version. Please update imports to:
    `from oscura.core.exceptions import ...`

Example:
    Old (deprecated)::

        from oscura.exceptions import OscuraError

    New (preferred)::

        from oscura.core.exceptions import OscuraError
"""

from __future__ import annotations

import warnings

# Emit deprecation warning
warnings.warn(
    "oscura.exceptions is deprecated and will be removed in a future version. "
    "Please use oscura.core.exceptions instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all exceptions from core module for backward compatibility
from oscura.core.exceptions import (  # noqa: E402
    AnalysisError,
    ConfigurationError,
    ExportError,
    FormatError,
    InsufficientDataError,
    LoaderError,
    OscuraError,
    SampleRateError,
    SecurityError,
    UnsupportedFormatError,
    ValidationError,
)

__all__ = [
    "AnalysisError",
    "ConfigurationError",
    "ExportError",
    "FormatError",
    "InsufficientDataError",
    "LoaderError",
    "OscuraError",
    "SampleRateError",
    "SecurityError",
    "UnsupportedFormatError",
    "ValidationError",
]
