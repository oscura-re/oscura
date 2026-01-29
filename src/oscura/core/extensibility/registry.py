"""Algorithm registry for custom algorithm injection.

This module implements a registry pattern that allows users to register
custom algorithms and implementations at extension points throughout Oscura.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class AlgorithmRegistry:
    """Singleton registry for custom algorithm implementations.

    Allows users to register custom algorithms for various categories
    (edge detection, peak finding, window functions, etc.) that can be
    used throughout Oscura.

    The registry validates algorithm signatures on registration and provides
    lookup by category and name.

    Example:
        >>> import oscura as osc
        >>> def my_edge_detector(data, threshold=0.5):
        ...     '''Custom Schmitt trigger edge detector'''
        ...     edges = []
        ...     state = data[0] > threshold
        ...     for i, val in enumerate(data):
        ...         new_state = val > threshold
        ...         if new_state != state:
        ...             edges.append(i)
        ...             state = new_state
        ...     return edges
        >>> # Register in algorithm registry
        >>> osc.register_algorithm('my_schmitt', my_edge_detector, category='edge_detector')
        >>> # Use custom algorithm
        >>> edges = osc.find_edges(trace, method='my_schmitt', threshold=0.7)

    Advanced Example:
        >>> # Register custom window function
        >>> import numpy as np
        >>> def custom_window(n, alpha=0.5):
        ...     x = np.linspace(0, 1, n)
        ...     return 0.5 * (1 + np.cos(2 * np.pi * alpha * (x - 0.5)))
        >>> osc.register_algorithm('custom_tukey', custom_window, category='window_func')
        >>> # Use in FFT
        >>> result = osc.fft(trace, nfft=8192, window='custom_tukey', alpha=0.3)
        >>> # List available algorithms
        >>> available = osc.get_algorithms('window_func')

    References:
        API-006: Algorithm Override Hooks
        pytest plugin system
        https://docs.pytest.org/en/stable/how-to/writing_plugins.html
    """

    _instance: AlgorithmRegistry | None = None
    _registries: dict[str, dict[str, Callable[..., Any]]]

    def __new__(cls) -> AlgorithmRegistry:
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registries = {}
        return cls._instance

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        category: str,
        validate: bool = True,
    ) -> None:
        """Register a custom algorithm.

        Args:
            name: Unique name for the algorithm within its category.
            func: Callable implementing the algorithm.
            category: Category of algorithm (e.g., 'edge_detector', 'peak_finder').
            validate: Whether to validate function signature. Default True.

        Raises:
            ValueError: If name already exists in category.
            TypeError: If func is not callable or signature is invalid.

        Example:
            >>> def my_algorithm(data, param1=1.0, param2=2.0):
            ...     return data * param1 + param2
            >>> registry = AlgorithmRegistry()
            >>> registry.register('my_algo', my_algorithm, 'preprocessor')
        """
        if not callable(func):
            raise TypeError(f"Algorithm must be callable, got {type(func).__name__}")

        # Initialize category if needed
        if category not in self._registries:
            self._registries[category] = {}

        # Check for duplicates
        if name in self._registries[category]:
            raise ValueError(f"Algorithm '{name}' already registered in category '{category}'")

        # Validate signature if requested
        if validate:
            self._validate_signature(func, category)

        # Register algorithm
        self._registries[category][name] = func

    def get(self, category: str, name: str) -> Callable[..., Any]:
        """Get algorithm by category and name.

        Args:
            category: Algorithm category.
            name: Algorithm name.

        Returns:
            The registered algorithm function.

        Raises:
            KeyError: If category or name not found.

        Example:
            >>> registry = AlgorithmRegistry()
            >>> edge_detector = registry.get('edge_detector', 'my_schmitt')
            >>> edges = edge_detector(data, threshold=0.5)
        """
        if category not in self._registries:
            raise KeyError(
                f"Category '{category}' not found. Available: {list(self._registries.keys())}"
            )

        if name not in self._registries[category]:
            raise KeyError(
                f"Algorithm '{name}' not found in category '{category}'. "
                f"Available: {list(self._registries[category].keys())}"
            )

        return self._registries[category][name]

    def list_categories(self) -> list[str]:
        """List all registered algorithm categories.

        Returns:
            List of category names.

        Example:
            >>> registry = AlgorithmRegistry()
            >>> categories = registry.list_categories()
            >>> print(categories)
            ['edge_detector', 'peak_finder', 'window_func']
        """
        return list(self._registries.keys())

    def list_algorithms(self, category: str) -> list[str]:
        """List all algorithms in a category.

        Args:
            category: Algorithm category.

        Returns:
            List of algorithm names in that category.

        Raises:
            KeyError: If category not found.

        Example:
            >>> registry = AlgorithmRegistry()
            >>> algorithms = registry.list_algorithms('edge_detector')
            >>> print(algorithms)
            ['threshold', 'hysteresis', 'my_schmitt']
        """
        if category not in self._registries:
            raise KeyError(
                f"Category '{category}' not found. Available: {list(self._registries.keys())}"
            )

        return list(self._registries[category].keys())

    def has_algorithm(self, category: str, name: str) -> bool:
        """Check if algorithm is registered.

        Args:
            category: Algorithm category.
            name: Algorithm name.

        Returns:
            True if algorithm is registered.

        Example:
            >>> if registry.has_algorithm('edge_detector', 'my_schmitt'):
            ...     detector = registry.get('edge_detector', 'my_schmitt')
        """
        return category in self._registries and name in self._registries[category]

    def unregister(self, category: str, name: str) -> None:
        """Remove algorithm from registry.

        Args:
            category: Algorithm category.
            name: Algorithm name.

        Raises:
            KeyError: If algorithm not found.

        Example:
            >>> registry.unregister('edge_detector', 'my_schmitt')
        """
        if not self.has_algorithm(category, name):
            raise KeyError(f"Algorithm '{name}' not found in category '{category}'")

        del self._registries[category][name]

    def clear_category(self, category: str) -> None:
        """Clear all algorithms in a category.

        Args:
            category: Category to clear.

        Example:
            >>> registry.clear_category('edge_detector')
        """
        if category in self._registries:
            self._registries[category].clear()

    def clear_all(self) -> None:
        """Clear all registered algorithms.

        Example:
            >>> registry.clear_all()
        """
        self._registries.clear()

    def _validate_signature(self, func: Callable[..., Any], category: str) -> None:
        """Validate function signature for category.

        Args:
            func: Function to validate.
            category: Category to validate against.

        Raises:
            TypeError: If signature is invalid for category.
        """
        sig = inspect.signature(func)

        # Check that function accepts **kwargs for extensibility
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        if not has_var_keyword:
            # Check if function has at least one parameter
            if len(sig.parameters) == 0:
                raise TypeError(
                    f"Algorithm function must accept at least one parameter "
                    f"(got {func.__name__} with no parameters)"
                )

    def __repr__(self) -> str:
        """String representation of registry."""
        total_algos = sum(len(algos) for algos in self._registries.values())
        return f"AlgorithmRegistry(categories={len(self._registries)}, algorithms={total_algos})"


# Global registry instance
_registry = AlgorithmRegistry()


def register_algorithm(
    name: str,
    func: Callable[..., Any],
    category: str,
    validate: bool = True,
) -> None:
    """Register a custom algorithm in the global registry.

    Convenience function for registering algorithms without accessing
    the registry instance directly.

    Args:
        name: Unique name for the algorithm.
        func: Callable implementing the algorithm.
        category: Algorithm category.
        validate: Whether to validate signature. Default True.

    Example:
        >>> import oscura as osc
        >>> def my_edge_detector(data, threshold=0.5):
        ...     return find_edges_custom(data, threshold)
        >>> osc.register_algorithm('my_edges', my_edge_detector, 'edge_detector')

    References:
        API-006: Algorithm Override Hooks
    """
    _registry.register(name, func, category, validate)


def get_algorithm(category: str, name: str) -> Callable[..., Any]:
    """Get algorithm from global registry.

    Args:
        category: Algorithm category.
        name: Algorithm name.

    Returns:
        The registered algorithm function.

    Example:
        >>> edge_detector = osc.get_algorithm('edge_detector', 'my_edges')
        >>> edges = edge_detector(data)

    References:
        API-006: Algorithm Override Hooks
    """
    return _registry.get(category, name)


def get_algorithms(category: str) -> list[str]:
    """List all algorithms in a category from global registry.

    Args:
        category: Algorithm category.

    Returns:
        List of algorithm names.

    Example:
        >>> available = osc.get_algorithms('window_func')
        >>> print(f"Available windows: {available}")

    References:
        API-006: Algorithm Override Hooks
    """
    return _registry.list_algorithms(category)


__all__ = [
    "AlgorithmRegistry",
    "get_algorithm",
    "get_algorithms",
    "register_algorithm",
]
