"""Base classes for trace transformations and pipeline stages.

This module implements the foundational abstract base classes for creating
custom trace transformations compatible with the Pipeline architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


class TraceTransformer(ABC):
    """Abstract base class for trace transformations.

    All pipeline stages and custom transformations must inherit from this class.
    Provides the fit/transform pattern similar to sklearn transformers.

    The TraceTransformer enforces a consistent interface:
    - transform(trace) -> trace: Required transformation method
    - fit(trace) -> self: Optional learning/calibration method
    - fit_transform(trace) -> trace: Convenience method
    - get_params() / set_params(): Hyperparameter access
    - clone(): Create a copy of the transformer

    Example:
        >>> class AmplitudeScaler(TraceTransformer):
        ...     def __init__(self, scale_factor=1.0):
        ...         self.scale_factor = scale_factor
        ...
        ...     def transform(self, trace):
        ...         scaled_data = trace.data * self.scale_factor
        ...         return WaveformTrace(
        ...             data=scaled_data,
        ...             metadata=trace.metadata
        ...         )
        ...
        >>> scaler = AmplitudeScaler(scale_factor=2.0)
        >>> result = scaler.transform(trace)

    References:
        API-004: TraceTransformer Base Class
        sklearn.base.BaseEstimator, TransformerMixin
    """

    @abstractmethod
    def transform(self, trace: WaveformTrace) -> WaveformTrace:
        """Transform a trace.

        Args:
            trace: Input WaveformTrace to transform.

        Returns:
            Transformed WaveformTrace.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement transform() method")

    def fit(self, trace: WaveformTrace) -> TraceTransformer:
        """Fit transformer to a reference trace (optional for stateful transformers).

        This method is optional and should be overridden by stateful transformers
        that need to learn parameters from a reference trace (e.g., normalization
        statistics, adaptive filters).

        Args:
            trace: Reference WaveformTrace to fit to.

        Returns:
            Self for method chaining.

        Example:
            >>> class AdaptiveNormalizer(TraceTransformer):
            ...     def __init__(self):
            ...         self.mean_ = None
            ...         self.std_ = None
            ...
            ...     def fit(self, trace):
            ...         self.mean_ = trace.data.mean()
            ...         self.std_ = trace.data.std()
            ...         return self
            ...
            ...     def transform(self, trace):
            ...         normalized = (trace.data - self.mean_) / self.std_
            ...         return WaveformTrace(
            ...             data=normalized,
            ...             metadata=trace.metadata
            ...         )
        """
        # Default implementation: no fitting required
        return self

    def fit_transform(self, trace: WaveformTrace) -> WaveformTrace:
        """Fit to trace, then transform it.

        Convenience method that calls fit() followed by transform().

        Args:
            trace: Input WaveformTrace to fit and transform.

        Returns:
            Transformed WaveformTrace.

        Example:
            >>> normalizer = AdaptiveNormalizer()
            >>> result = normalizer.fit_transform(reference_trace)
        """
        return self.fit(trace).transform(trace)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this transformer.

        Args:
            deep: If True, will return parameters for nested objects.

        Returns:
            Dictionary of parameter names mapped to their values.

        Example:
            >>> scaler = AmplitudeScaler(scale_factor=2.0)
            >>> params = scaler.get_params()
            >>> print(params)
            {'scale_factor': 2.0}
        """
        params = {}
        for key in dir(self):
            # Skip private/magic attributes and methods
            if key.startswith("_") or callable(getattr(self, key)):
                continue
            value = getattr(self, key)
            params[key] = value

            # Handle nested transformers if deep=True
            if deep and hasattr(value, "get_params"):
                nested_params = value.get_params(deep=True)
                for nested_key, nested_value in nested_params.items():
                    params[f"{key}__{nested_key}"] = nested_value

        return params

    def set_params(self, **params: Any) -> TraceTransformer:
        """Set parameters for this transformer.

        Args:
            **params: Parameter names and values to set.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If parameter name is invalid.

        Example:
            >>> scaler = AmplitudeScaler(scale_factor=1.0)
            >>> scaler.set_params(scale_factor=3.0)
            >>> print(scaler.scale_factor)
            3.0
        """
        if not params:
            return self

        valid_params = self.get_params(deep=False)

        for key, value in params.items():
            # Handle nested parameters (e.g., 'filter__cutoff')
            if "__" in key:
                nested_obj, nested_key = key.split("__", 1)
                if nested_obj not in valid_params:
                    raise ValueError(
                        f"Invalid parameter {nested_obj} for transformer {self.__class__.__name__}"
                    )
                nested = getattr(self, nested_obj)
                if hasattr(nested, "set_params"):
                    nested.set_params(**{nested_key: value})
                else:
                    raise ValueError(f"Parameter {nested_obj} does not support set_params")
            else:
                if key not in valid_params:
                    raise ValueError(
                        f"Invalid parameter {key} for transformer "
                        f"{self.__class__.__name__}. "
                        f"Valid parameters: {list(valid_params.keys())}"
                    )
                setattr(self, key, value)

        return self

    def clone(self) -> TraceTransformer:
        """Create a copy of this transformer with the same parameters.

        Returns:
            New instance of the transformer with same parameters.

        Example:
            >>> scaler = AmplitudeScaler(scale_factor=2.0)
            >>> scaler_copy = scaler.clone()
            >>> scaler_copy.scale_factor
            2.0
        """
        params = self.get_params(deep=False)
        return self.__class__(**params)

    def __getstate__(self) -> dict[str, Any]:
        """Get state for pickling.

        Returns:
            Dictionary containing transformer state.
        """
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set state from unpickling.

        Args:
            state: Dictionary containing transformer state.
        """
        self.__dict__.update(state)

    def get_intermediate_result(self, key: str) -> Any:
        """Get intermediate result from last transformation.

        Some transformers cache intermediate results (e.g., FFT coefficients,
        filter states) that can be accessed after transformation.

        Args:
            key: Name of intermediate result to retrieve.

        Returns:
            Intermediate result value.

        Raises:
            KeyError: If key not found or transformer doesn't support intermediates.

        Example:
            >>> filter = LowPassFilter(cutoff=1e6)
            >>> result = filter.transform(trace)
            >>> transfer_func = filter.get_intermediate_result('transfer_function')

        References:
            API-005: Intermediate Result Access
        """
        # Check if transformer has _intermediates cache
        if not hasattr(self, "_intermediates"):
            raise KeyError(f"{self.__class__.__name__} does not cache intermediate results")

        intermediates = self._intermediates
        if key not in intermediates:
            available = list(intermediates.keys())
            raise KeyError(
                f"Intermediate '{key}' not found in {self.__class__.__name__}. "
                f"Available: {available}"
            )

        return intermediates[key]

    def has_intermediate_result(self, key: str) -> bool:
        """Check if intermediate result is available.

        Args:
            key: Name of intermediate result.

        Returns:
            True if intermediate result exists.

        Example:
            >>> if filter.has_intermediate_result('impulse_response'):
            ...     impulse = filter.get_intermediate_result('impulse_response')

        References:
            API-005: Intermediate Result Access
        """
        if not hasattr(self, "_intermediates"):
            return False
        return key in self._intermediates

    def list_intermediate_results(self) -> list[str]:
        """List all available intermediate result keys.

        Returns:
            List of intermediate result names, or empty list if none available.

        Example:
            >>> print(filter.list_intermediate_results())
            ['transfer_function', 'impulse_response', 'frequency_response']

        References:
            API-005: Intermediate Result Access
        """
        if not hasattr(self, "_intermediates"):
            return []
        return list(self._intermediates.keys())

    def _cache_intermediate(self, key: str, value: Any) -> None:
        """Cache an intermediate result for later access.

        This is a protected method for subclasses to use when storing
        intermediate computation results.

        Args:
            key: Name of intermediate result.
            value: Value to cache.

        Example (in subclass):
            >>> def transform(self, trace):
            ...     fft_coeffs = compute_fft(trace)
            ...     self._cache_intermediate('fft_coeffs', fft_coeffs)
            ...     return processed_trace

        References:
            API-005: Intermediate Result Access
        """
        if not hasattr(self, "_intermediates"):
            self._intermediates = {}
        self._intermediates[key] = value

    def _clear_intermediates(self) -> None:
        """Clear all cached intermediate results.

        Useful for freeing memory when intermediate results are no longer needed.

        Example (in subclass):
            >>> def transform(self, trace):
            ...     self._clear_intermediates()  # Clear previous results
            ...     # ... perform transformation ...

        References:
            API-005: Intermediate Result Access
        """
        if hasattr(self, "_intermediates"):
            self._intermediates.clear()


__all__ = ["TraceTransformer"]
