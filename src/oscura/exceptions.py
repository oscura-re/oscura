"""Oscura exception hierarchy - DEPRECATED compatibility module.

.. deprecated:: 1.0.0
    This module is deprecated for backward compatibility only.
    New code MUST import from `oscura.core.exceptions` directly.
    This module will be removed in a future major version.

This module re-exports exceptions from oscura.core.exceptions.
The canonical location for all exception classes is `oscura.core.exceptions`.

Why two files exist:
    - `oscura/core/exceptions.py`: Canonical implementation of all exception classes
    - `oscura/exceptions.py` (this file): Deprecated re-export for backward compatibility

Migration guide:
    Old (deprecated):
        from oscura.exceptions import LoaderError

    New (preferred):
        from oscura.core.exceptions import LoaderError
"""

import warnings

# Issue deprecation warning on import
warnings.warn(
    "oscura.exceptions is deprecated. "
    "Import from oscura.core.exceptions instead. "
    "This module will be removed in a future major version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all exceptions from core.exceptions
from oscura.core.exceptions import (  # noqa: E402
    AnalysisError,
    ConfigurationError,
    ExportError,
    FormatError,
    InsufficientDataError,
    LoaderError,
    OscuraError,
    SampleRateError,
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
    "UnsupportedFormatError",
    "ValidationError",
]
