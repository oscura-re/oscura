"""Lazy evaluation module for deferred computation in Oscura workflows.

This module provides lazy evaluation primitives that defer computation until
results are actually accessed. Designed for analysis workflows where not all
intermediate results may be needed, reducing memory usage and computation time.

Key features:
- Thread-safe lazy evaluation with compute-once semantics
- Chained operations without eager evaluation
- Partial evaluation (compute subset of results)
- Memory-efficient release of source data after computation
- Integration with Oscura's memory monitoring and progress tracking


Example:
    >>> from oscura.core.lazy import LazyResult, lazy, LazyDict
    >>>
    >>> # Defer expensive FFT computation
    >>> @lazy
    >>> def compute_fft(signal, nfft):
    ...     return np.fft.fft(signal, n=nfft)
    >>>
    >>> # Create lazy result - not computed yet
    >>> lazy_fft = compute_fft(signal, 8192)
    >>> print(lazy_fft.is_computed())  # False
    >>>
    >>> # Access triggers computation
    >>> spectrum = lazy_fft.value  # Computed now
    >>> print(lazy_fft.is_computed())  # True
    >>>
    >>> # Multiple accesses use cached result
    >>> spectrum2 = lazy_fft.value  # Returns cached value
    >>>
    >>> # LazyDict for multiple lazy results
    >>> results = LazyDict()
    >>> results['fft'] = LazyResult(lambda: np.fft.fft(signal, 8192))
    >>> results['power'] = LazyResult(lambda: np.abs(results['fft'].value)**2)
    >>> # Access triggers computation chain
    >>> power_spectrum = results['power']  # Computes fft, then power

References:
    Python lazy evaluation patterns
    Threading locks for thread-safe computation
    Oscura memory monitoring (core.memory_monitor)
"""

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class LazyComputeStats:
    """Statistics for lazy computation tracking.

    Attributes:
        total_created: Total number of LazyResult instances created.
        total_computed: Number of LazyResult instances that have been computed.
        total_invalidated: Number of invalidations (for delta analysis).
        compute_time_total: Total time spent computing (seconds).
        cache_hits: Number of times a computed result was reused.

    Example:
        >>> stats = LazyComputeStats()
        >>> stats.total_created += 1
        >>> stats.total_computed += 1
        >>> print(f"Computed: {stats.total_computed}/{stats.total_created}")
    """

    total_created: int = 0
    total_computed: int = 0
    total_invalidated: int = 0
    compute_time_total: float = 0.0
    cache_hits: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Fraction of accesses that were cache hits (0.0-1.0).
        """
        total_accesses = self.total_computed + self.cache_hits
        if total_accesses == 0:
            return 0.0
        return self.cache_hits / total_accesses

    def __str__(self) -> str:
        """Format statistics as readable string."""
        return (
            f"Lazy Computation Statistics:\n"
            f"  Created: {self.total_created}\n"
            f"  Computed: {self.total_computed}\n"
            f"  Cache Hits: {self.cache_hits}\n"
            f"  Hit Rate: {self.hit_rate * 100:.1f}%\n"
            f"  Invalidations: {self.total_invalidated}\n"
            f"  Total Compute Time: {self.compute_time_total:.3f}s\n"
        )


# Global statistics tracker
_global_stats = LazyComputeStats()
_stats_lock = threading.Lock()


def get_lazy_stats() -> LazyComputeStats:
    """Get global lazy computation statistics.

    Returns:
        Global LazyComputeStats instance.

    Example:
        >>> stats = get_lazy_stats()
        >>> print(stats)
        Lazy Computation Statistics:
          Created: 42
          Computed: 35
          ...

    References:
        MEM-031: Cache statistics tracking
    """
    with _stats_lock:
        return LazyComputeStats(
            total_created=_global_stats.total_created,
            total_computed=_global_stats.total_computed,
            total_invalidated=_global_stats.total_invalidated,
            compute_time_total=_global_stats.compute_time_total,
            cache_hits=_global_stats.cache_hits,
        )


def reset_lazy_stats() -> None:
    """Reset global lazy computation statistics.

    Example:
        >>> reset_lazy_stats()
        >>> stats = get_lazy_stats()
        >>> assert stats.total_created == 0
    """
    global _global_stats
    with _stats_lock:
        _global_stats = LazyComputeStats()


class LazyResult(Generic[T]):
    """Deferred computation wrapper with thread-safe compute-once semantics.

    Wraps a computation function that will be called only when the result is
    first accessed. Subsequent accesses return the cached result. Thread-safe
    for parallel access from multiple analyzers.


    Attributes:
        name: Optional name for debugging/logging.

    Example:
        >>> # Create lazy FFT computation
        >>> lazy_fft = LazyResult(
        ...     lambda: np.fft.fft(signal, n=8192),
        ...     name="fft_8192"
        ... )
        >>>
        >>> # Check if computed without triggering computation
        >>> if not lazy_fft.is_computed():
        ...     print("Not computed yet")
        >>>
        >>> # Access triggers computation
        >>> spectrum = lazy_fft.value
        >>>
        >>> # Subsequent accesses use cache
        >>> spectrum2 = lazy_fft.value  # No recomputation
        >>>
        >>> # Invalidate for delta analysis
        >>> lazy_fft.invalidate()
        >>> spectrum3 = lazy_fft.value  # Recomputes

    References:
        Python threading for thread safety
        PERF-002: Lazy evaluation requirements
    """

    def __init__(
        self,
        compute_fn: Callable[[], T],
        name: str = "",
        *,
        weak_source: bool = False,
    ):
        """Initialize lazy result.

        Args:
            compute_fn: Function to call to compute the result.
            name: Optional name for debugging/logging.
            weak_source: If True, use weak reference to source data
                (allows GC after computation).

        Example:
            >>> lazy_result = LazyResult(
            ...     lambda: expensive_computation(),
            ...     name="expensive_op"
            ... )
        """
        self._compute_fn = compute_fn
        self._name = name or f"LazyResult_{id(self)}"
        self._result: T | None = None
        self._computed = False
        self._lock = threading.RLock()
        self._weak_source = weak_source
        self._source_released = False

        # Track creation
        with _stats_lock:
            _global_stats.total_created += 1

    @property
    def value(self) -> T:
        """Get the result, computing if necessary.

        Thread-safe lazy computation with compute-once semantics.
        Multiple concurrent accesses will only compute once.

        Returns:
            The computed result.

        Raises:
            Exception: Any exception raised by the compute function.

        Example:
            >>> lazy_fft = LazyResult(lambda: np.fft.fft(signal))
            >>> spectrum = lazy_fft.value  # Computes here
            >>> spectrum2 = lazy_fft.value  # Uses cache

        References:
            PERF-002: Lazy evaluation for analysis pipelines
        """
        with self._lock:
            if self._computed:
                # Cache hit
                with _stats_lock:
                    _global_stats.cache_hits += 1
                return self._result  # type: ignore[return-value]

            # Compute result
            import time

            start_time = time.time()

            try:
                self._result = self._compute_fn()
                self._computed = True

                # Track computation
                compute_time = time.time() - start_time
                with _stats_lock:
                    _global_stats.total_computed += 1
                    _global_stats.compute_time_total += compute_time

                # Optionally release source data
                if self._weak_source and not self._source_released:
                    self._release_source()

                return self._result

            except Exception:
                # Don't cache errors, allow retry
                self._computed = False
                raise

    def is_computed(self) -> bool:
        """Check if result has been computed without triggering computation.

        Returns:
            True if result is computed and cached.

        Example:
            >>> lazy_result = LazyResult(lambda: expensive_op())
            >>> if lazy_result.is_computed():
            ...     result = lazy_result.value  # No computation
            ... else:
            ...     print("Will compute on next access")

        References:
            API-012: Lazy result access patterns
        """
        with self._lock:
            return self._computed

    def invalidate(self) -> None:
        """Mark result as invalid, forcing recomputation on next access.

        Useful for delta analysis where the underlying data changes and
        results need to be recomputed.

        Example:
            >>> lazy_fft = LazyResult(lambda: np.fft.fft(signal))
            >>> spectrum1 = lazy_fft.value
            >>>
            >>> # Signal data changed
            >>> signal = new_signal
            >>> lazy_fft.invalidate()
            >>>
            >>> # Next access recomputes with new signal
            >>> spectrum2 = lazy_fft.value

        References:
            API-012: Delta analysis support
        """
        with self._lock:
            self._computed = False
            self._result = None
            self._source_released = False

            with _stats_lock:
                _global_stats.total_invalidated += 1

    def get_if_computed(self) -> T | None:
        """Get result only if already computed, otherwise return None.

        Returns:
            Computed result or None if not yet computed.

        Example:
            >>> lazy_result = LazyResult(lambda: expensive_op())
            >>> result = lazy_result.get_if_computed()  # None
            >>> _ = lazy_result.value  # Compute
            >>> result = lazy_result.get_if_computed()  # Returns result
        """
        with self._lock:
            if self._computed:
                return self._result
            return None

    def peek(self) -> tuple[bool, T | None]:
        """Get computation status and result if available.

        Returns:
            Tuple of (is_computed, result). Result is None if not computed.

        Example:
            >>> lazy_result = LazyResult(lambda: expensive_op())
            >>> computed, result = lazy_result.peek()
            >>> if computed:
            ...     print(f"Result: {result}")
            ... else:
            ...     print("Not computed yet")
        """
        with self._lock:
            return (self._computed, self._result)

    def map(self, fn: Callable[[T], Any]) -> LazyResult[Any]:
        """Create a new lazy result by applying a function to this result.

        Enables chained lazy operations without eager evaluation.

        Args:
            fn: Function to apply to the result.

        Returns:
            New LazyResult that computes fn(self.value).

        Example:
            >>> lazy_fft = LazyResult(lambda: np.fft.fft(signal))
            >>> lazy_power = lazy_fft.map(lambda x: np.abs(x)**2)
            >>> lazy_peak = lazy_power.map(lambda x: x.max())
            >>>
            >>> # Nothing computed yet
            >>> peak = lazy_peak.value  # Computes entire chain

        References:
            PERF-002: Lazy evaluation for chained operations
        """
        return LazyResult(
            lambda: fn(self.value),
            name=f"{self._name}.map({fn.__name__})",
            weak_source=self._weak_source,
        )

    def _release_source(self) -> None:
        """Release source data to allow garbage collection.

        After computation completes, we can release the source data
        if weak_source=True was specified. This replaces the compute
        function's closure to break references to large input data.

        Note: We don't call gc.collect() here as it would be called
        very frequently and is expensive. Python's automatic GC will
        handle cleanup.
        """
        # Clear the compute function's closure to release references
        if hasattr(self._compute_fn, "__closure__") and self._compute_fn.__closure__:
            # Can't directly clear closure, but can replace function
            # to break references
            result = self._result

            def return_result() -> T:
                return result  # type: ignore[return-value]

            self._compute_fn = return_result

        self._source_released = True
        # Let Python's automatic GC handle cleanup

    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "computed" if self._computed else "deferred"
        return f"LazyResult(name={self._name!r}, status={status})"


class LazyDict(dict[str, Any]):
    """Dictionary where LazyResult values are auto-evaluated on access.

    Extends standard dict to automatically evaluate LazyResult values when
    accessed. Regular (non-lazy) values pass through unchanged.

    Useful for collections of analysis results where some may not be needed.

    Example:
        >>> results = LazyDict()
        >>> results['fft'] = LazyResult(lambda: np.fft.fft(signal))
        >>> results['power'] = LazyResult(lambda: np.abs(results['fft'])**2)
        >>> results['constant'] = 42  # Non-lazy value
        >>>
        >>> # Access auto-evaluates lazy results
        >>> fft_spectrum = results['fft']  # Computes FFT
        >>> power_spectrum = results['power']  # Computes power
        >>> const = results['constant']  # Returns 42 directly
        >>>
        >>> # Check if computed without triggering computation
        >>> fft_lazy = super(LazyDict, results).__getitem__('fft')
        >>> if fft_lazy.is_computed():
        ...     print("FFT already computed")

    References:
        API-012: Lazy result access patterns
    """

    def __getitem__(self, key: str) -> Any:
        """Get value, auto-evaluating if it's a LazyResult.

        Args:
            key: Dictionary key.

        Returns:
            Evaluated value (LazyResult.value) or raw value.

        Example:
            >>> lazy_dict = LazyDict()
            >>> lazy_dict['result'] = LazyResult(lambda: expensive_op())
            >>> value = lazy_dict['result']  # Auto-evaluates
        """
        value = super().__getitem__(key)
        if isinstance(value, LazyResult):
            return value.value
        return value

    def get_lazy(self, key: str) -> LazyResult[Any] | Any:
        """Get the raw value without auto-evaluation.

        Returns the LazyResult instance itself, not its value.

        Args:
            key: Dictionary key.

        Returns:
            Raw value (may be LazyResult instance).

        Example:
            >>> lazy_dict = LazyDict()
            >>> lazy_dict['result'] = LazyResult(lambda: expensive_op())
            >>> lazy_obj = lazy_dict.get_lazy('result')
            >>> if not lazy_obj.is_computed():
            ...     print("Will compute on access")
        """
        return super().__getitem__(key)

    def is_computed(self, key: str) -> bool:
        """Check if a lazy value has been computed.

        Args:
            key: Dictionary key.

        Returns:
            True if value is computed (or not lazy), False otherwise.

        Example:
            >>> if not lazy_dict.is_computed('fft'):
            ...     print("FFT not computed yet")
        """
        value = super().__getitem__(key)
        if isinstance(value, LazyResult):
            return value.is_computed()
        return True  # Non-lazy values are "computed"

    def invalidate(self, key: str) -> None:
        """Invalidate a lazy result, forcing recomputation.

        Args:
            key: Dictionary key.

        Example:
            >>> lazy_dict.invalidate('fft')
            >>> # Next access will recompute
            >>> fft = lazy_dict['fft']
        """
        value = super().__getitem__(key)
        if isinstance(value, LazyResult):
            value.invalidate()

    def invalidate_all(self) -> None:
        """Invalidate all lazy results in the dictionary.

        Example:
            >>> lazy_dict.invalidate_all()
            >>> # All lazy values will recompute on next access
        """
        for value in self.values():
            if isinstance(value, LazyResult):
                value.invalidate()

    def computed_keys(self) -> list[str]:
        """Get list of keys with computed values.

        Returns:
            List of keys whose values are computed.

        Example:
            >>> computed = lazy_dict.computed_keys()
            >>> print(f"Computed: {computed}")
        """
        return [
            key
            for key, value in super().items()
            if not isinstance(value, LazyResult) or value.is_computed()
        ]

    def deferred_keys(self) -> list[str]:
        """Get list of keys with deferred (not computed) values.

        Returns:
            List of keys whose LazyResult values are not computed.

        Example:
            >>> deferred = lazy_dict.deferred_keys()
            >>> print(f"Not computed: {deferred}")
        """
        return [
            key
            for key, value in super().items()
            if isinstance(value, LazyResult) and not value.is_computed()
        ]


def lazy(fn: Callable[..., T]) -> Callable[..., LazyResult[T]]:
    """Decorator to make a function return a LazyResult.

    Wraps a function so it returns a LazyResult instead of computing
    immediately. Useful for expensive analysis functions.

    Args:
        fn: Function to wrap.

    Returns:
        Wrapped function that returns LazyResult.

    Example:
        >>> @lazy
        ... def compute_fft(signal, nfft):
        ...     print("Computing FFT...")
        ...     return np.fft.fft(signal, n=nfft)
        >>>
        >>> # Returns LazyResult, doesn't compute yet
        >>> lazy_fft = compute_fft(signal, 8192)
        >>> print("Created lazy result")
        >>>
        >>> # Access triggers computation
        >>> spectrum = lazy_fft.value
        >>> # Prints: "Computing FFT..."
        >>>
        >>> # Second access uses cache
        >>> spectrum2 = lazy_fft.value
        >>> # No print - uses cached result

    References:
        PERF-002: Lazy evaluation for analysis pipelines
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> LazyResult[T]:
        def compute() -> T:
            return fn(*args, **kwargs)

        return LazyResult(
            compute,
            name=fn.__name__,
        )

    return wrapper


class LazyAnalysisResult:
    """Lazy wrapper for multi-domain analysis results.

    Provides lazy evaluation for analysis engines that produce results across
    multiple domains (time, frequency, statistical, etc.). Only computes
    results for domains that are actually accessed.


    Attributes:
        domains: List of available analysis domains.

    Example:
        >>> # Create analyzer with multiple domains
        >>> analyzer = SignalAnalyzer()
        >>>
        >>> # Wrap in lazy result - nothing computed yet
        >>> lazy_results = LazyAnalysisResult(
        ...     analyzer,
        ...     signal_data,
        ...     domains=['time', 'frequency', 'statistics']
        ... )
        >>>
        >>> # Only compute frequency domain
        >>> freq_results = lazy_results.get_domain('frequency')
        >>> # Time and statistics domains not computed
        >>>
        >>> # Check what's been computed
        >>> print(lazy_results.computed_domains())  # ['frequency']
        >>> print(lazy_results.deferred_domains())  # ['time', 'statistics']
        >>>
        >>> # Access multiple domains
        >>> all_results = lazy_results.compute_all()

    References:
        PERF-002: Lazy evaluation requirements
        API-012: Multi-domain analysis patterns
    """

    def __init__(
        self,
        engine: Any,
        data: Any,
        domains: list[str],
        *,
        compute_fn_template: Callable[[Any, Any, str], Any] | None = None,
    ):
        """Initialize lazy analysis result.

        Args:
            engine: Analysis engine instance.
            data: Input data for analysis.
            domains: List of available analysis domains.
            compute_fn_template: Optional custom compute function.
                Signature: fn(engine, data, domain) -> result.
                Default uses engine.analyze(data, domain=domain).

        Example:
            >>> lazy_results = LazyAnalysisResult(
            ...     my_analyzer,
            ...     signal_data,
            ...     domains=['time', 'frequency', 'wavelet']
            ... )
        """
        self._engine = engine
        self._data = data
        self.domains = domains
        self._compute_fn = compute_fn_template or self._default_compute

        # Create lazy results for each domain
        self._domain_results = LazyDict()
        for domain in domains:

            def make_compute_fn(d: str = domain) -> Callable[[], Any]:
                def compute_domain() -> Any:
                    return self._compute_fn(self._engine, self._data, d)

                return compute_domain

            self._domain_results[domain] = LazyResult(
                make_compute_fn(),
                name=f"{engine.__class__.__name__}.{domain}",
            )

    def _default_compute(self, engine: Any, data: Any, domain: str) -> dict[str, Any]:
        """Default compute function for domain analysis.

        Args:
            engine: Analysis engine.
            data: Input data.
            domain: Domain to analyze.

        Returns:
            Analysis result for domain.

        Raises:
            AttributeError: If engine has no analyze() or analyze_{domain}() method.
        """
        # Try common patterns
        if hasattr(engine, "analyze"):
            result: dict[str, Any] = engine.analyze(data, domain=domain)
            return result
        elif hasattr(engine, f"analyze_{domain}"):
            method = getattr(engine, f"analyze_{domain}")
            result = method(data)
            return result
        else:
            raise AttributeError(
                f"Engine {engine.__class__.__name__} has no analyze() method "
                f"or analyze_{domain}() method"
            )

    def get_domain(self, domain: str) -> Any:
        """Get results for specific domain, computing only that domain.

        Args:
            domain: Domain name (e.g., 'time', 'frequency').

        Returns:
            Analysis results for the domain.

        Raises:
            KeyError: If domain not available.

        Example:
            >>> freq_results = lazy_results.get_domain('frequency')
            >>> # Only frequency domain computed

        References:
            PERF-002: Partial evaluation
        """
        if domain not in self.domains:
            raise KeyError(f"Domain '{domain}' not available. Available: {self.domains}")
        return self._domain_results[domain]

    def computed_domains(self) -> list[str]:
        """Get list of domains that have been computed.

        Returns:
            List of computed domain names.

        Example:
            >>> computed = lazy_results.computed_domains()
            >>> print(f"Computed: {computed}")
        """
        return self._domain_results.computed_keys()

    def deferred_domains(self) -> list[str]:
        """Get list of domains that have not been computed.

        Returns:
            List of deferred domain names.

        Example:
            >>> deferred = lazy_results.deferred_domains()
            >>> print(f"Not computed: {deferred}")
        """
        return self._domain_results.deferred_keys()

    def compute_all(self) -> dict[str, Any]:
        """Compute all domains and return results dictionary.

        Returns:
            Dictionary mapping domain names to results.

        Example:
            >>> all_results = lazy_results.compute_all()
            >>> print(all_results.keys())  # All domains

        References:
            API-012: Bulk computation
        """
        return {domain: self.get_domain(domain) for domain in self.domains}

    def invalidate_domain(self, domain: str) -> None:
        """Invalidate a specific domain's results.

        Args:
            domain: Domain to invalidate.

        Example:
            >>> lazy_results.invalidate_domain('frequency')
            >>> # Next access will recompute
        """
        self._domain_results.invalidate(domain)

    def invalidate_all(self) -> None:
        """Invalidate all domain results.

        Example:
            >>> lazy_results.invalidate_all()
            >>> # All domains will recompute on next access
        """
        self._domain_results.invalidate_all()

    def __getitem__(self, domain: str) -> Any:
        """Dictionary-style access to domains.

        Args:
            domain: Domain name.

        Returns:
            Domain results.

        Example:
            >>> freq_results = lazy_results['frequency']
        """
        return self.get_domain(domain)

    def __repr__(self) -> str:
        """String representation for debugging."""
        computed = self.computed_domains()
        deferred = self.deferred_domains()
        return (
            f"LazyAnalysisResult(domains={self.domains}, computed={computed}, deferred={deferred})"
        )


__all__ = [
    "LazyAnalysisResult",
    "LazyComputeStats",
    "LazyDict",
    "LazyResult",
    "get_lazy_stats",
    "lazy",
    "reset_lazy_stats",
]
