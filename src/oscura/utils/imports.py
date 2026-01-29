"""Lazy import utilities for optional dependencies.

This module provides utilities for handling optional dependencies gracefully,
with helpful error messages directing users to install missing extras.
"""

from typing import Any


class MissingOptionalDependency(ImportError):
    """Raised when an optional dependency is required but not installed."""


def require_matplotlib() -> Any:
    """Import and return matplotlib, or raise helpful error if not installed.

    Returns:
        matplotlib module

    Raises:
        MissingOptionalDependency: If matplotlib is not installed

    Example:
        >>> plt = require_matplotlib().pyplot
        >>> plt.plot([1, 2, 3])
    """
    try:
        import matplotlib

        return matplotlib
    except ImportError as e:
        raise MissingOptionalDependency(
            "Visualization features require matplotlib.\n\n"
            "Install with:\n"
            "  pip install oscura[visualization]    # Just matplotlib\n"
            "  pip install oscura[standard]         # Recommended for most users\n"
            "  pip install oscura[all]              # Everything\n\n"
            "Or install matplotlib directly:\n"
            "  pip install matplotlib\n"
        ) from e


def require_pandas() -> Any:
    """Import and return pandas, or raise helpful error if not installed.

    Returns:
        pandas module

    Raises:
        MissingOptionalDependency: If pandas is not installed

    Example:
        >>> pd = require_pandas()
        >>> df = pd.DataFrame({'a': [1, 2, 3]})
    """
    try:
        import pandas

        return pandas
    except ImportError as e:
        raise MissingOptionalDependency(
            "DataFrame features require pandas.\n\n"
            "Install with:\n"
            "  pip install oscura[dataframes]       # Pandas + Excel export\n"
            "  pip install oscura[standard]         # Recommended for most users\n"
            "  pip install oscura[all]              # Everything\n\n"
            "Or install pandas directly:\n"
            "  pip install pandas\n"
        ) from e


def require_psutil() -> Any:
    """Import and return psutil, or raise helpful error if not installed.

    Returns:
        psutil module

    Raises:
        MissingOptionalDependency: If psutil is not installed

    Example:
        >>> psutil = require_psutil()
        >>> psutil.virtual_memory()
    """
    try:
        import psutil

        return psutil
    except ImportError as e:
        raise MissingOptionalDependency(
            "System monitoring features require psutil.\n\n"
            "Install with:\n"
            "  pip install oscura[system]           # System utilities\n"
            "  pip install oscura[standard]         # Recommended for most users\n"
            "  pip install oscura[all]              # Everything\n\n"
            "Or install psutil directly:\n"
            "  pip install psutil\n"
        ) from e


def require_jinja2() -> Any:
    """Import and return jinja2, or raise helpful error if not installed.

    Returns:
        jinja2 module

    Raises:
        MissingOptionalDependency: If jinja2 is not installed

    Example:
        >>> jinja2 = require_jinja2()
        >>> template = jinja2.Template("Hello {{ name }}")
    """
    try:
        import jinja2

        return jinja2
    except ImportError as e:
        raise MissingOptionalDependency(
            "Report generation features require jinja2.\n\n"
            "Install with:\n"
            "  pip install oscura[reporting]        # Report generation\n"
            "  pip install oscura[standard]         # Recommended for most users\n"
            "  pip install oscura[all]              # Everything\n\n"
            "Or install jinja2 directly:\n"
            "  pip install jinja2\n"
        ) from e


# Convenience functions for checking availability without raising errors
def has_matplotlib() -> bool:
    """Check if matplotlib is available.

    Returns:
        True if matplotlib can be imported, False otherwise
    """
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


def has_pandas() -> bool:
    """Check if pandas is available.

    Returns:
        True if pandas can be imported, False otherwise
    """
    try:
        import pandas  # noqa: F401

        return True
    except ImportError:
        return False


def has_psutil() -> bool:
    """Check if psutil is available.

    Returns:
        True if psutil can be imported, False otherwise
    """
    try:
        import psutil  # noqa: F401

        return True
    except ImportError:
        return False


def has_jinja2() -> bool:
    """Check if jinja2 is available.

    Returns:
        True if jinja2 can be imported, False otherwise
    """
    try:
        import jinja2  # noqa: F401

        return True
    except ImportError:
        return False
