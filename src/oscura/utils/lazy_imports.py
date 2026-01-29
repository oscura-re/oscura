"""Lazy import utilities for optional heavy dependencies.

Delays import of heavy libraries (matplotlib, plotly, asammdf) until first use.
This significantly reduces import time for CLI and lightweight operations.

Example:
    >>> from oscura.utils.lazy_imports import lazy_import
    >>> plt = lazy_import('matplotlib.pyplot')
    >>> # matplotlib not imported yet
    >>> plt.figure()  # matplotlib imported on first use
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any


class LazyModule:
    """Lazy module loader that delays import until attribute access.

    Args:
        name: Module name to import.
        package: Package name for relative imports.

    Example:
        >>> plt = LazyModule('matplotlib.pyplot')
        >>> # No import yet
        >>> fig = plt.figure()  # Import happens here
    """

    def __init__(self, name: str, package: str | None = None) -> None:
        """Initialize lazy module loader.

        Args:
            name: Module name to import.
            package: Package name for relative imports.
        """
        self._module_name = name
        self._module_package = package
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        """Load the module if not already loaded.

        Returns:
            Loaded module.
        """
        if self._module is None:
            self._module = importlib.import_module(self._module_name, self._module_package)
        return self._module

    def __getattr__(self, name: str) -> Any:
        """Get attribute from lazy-loaded module.

        Args:
            name: Attribute name.

        Returns:
            Attribute value from loaded module.
        """
        return getattr(self._load(), name)

    def __dir__(self) -> list[str]:
        """Get directory of lazy-loaded module.

        Returns:
            List of attribute names.
        """
        return dir(self._load())


def lazy_import(name: str, package: str | None = None) -> LazyModule:
    """Create a lazy-loading module proxy.

    Module is not imported until first attribute access.
    Reduces startup time and memory usage.

    Args:
        name: Module name to import.
        package: Package name for relative imports.

    Returns:
        Lazy module proxy.

    Example:
        >>> plt = lazy_import('matplotlib.pyplot')
        >>> # matplotlib.pyplot not imported yet
        >>> fig = plt.figure()  # Import happens here
        >>> plt.show()
    """
    return LazyModule(name, package)


def is_available(module_name: str) -> bool:
    """Check if a module is available without importing it.

    Args:
        module_name: Module name to check.

    Returns:
        True if module can be imported.

    Example:
        >>> if is_available('matplotlib'):
        ...     import matplotlib.pyplot as plt
        ...     plt.figure()
    """
    if module_name in sys.modules:
        return True

    try:
        importlib.util.find_spec(module_name)  # type: ignore[attr-defined]
        return True
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
        return False


def require_module(module_name: str, feature: str = "") -> None:
    """Raise informative error if required module is not available.

    Args:
        module_name: Module name to check.
        feature: Feature name that requires the module (for error message).

    Raises:
        ImportError: If module is not available.

    Example:
        >>> require_module('matplotlib', 'plotting')
        >>> import matplotlib.pyplot as plt  # Safe to import now
    """
    if not is_available(module_name):
        feature_msg = f" for {feature}" if feature else ""
        raise ImportError(
            f"{module_name} is required{feature_msg}. Install with: pip install {module_name}"
        )


__all__ = [
    "LazyModule",
    "is_available",
    "lazy_import",
    "require_module",
]
