"""Functional composition operators for trace transformations.

This module implements compose() and pipe() functions for functional-style
trace processing, with support for operator overloading.
"""

from collections.abc import Callable
from functools import reduce, wraps
from typing import Any, TypeVar

from oscura.core.types import WaveformTrace

# Type variables for generic composition
T = TypeVar("T")
TraceFunc = Callable[[WaveformTrace], WaveformTrace]


def compose(*funcs: TraceFunc) -> TraceFunc:
    """Compose functions right-to-left: compose(f, g, h)(x) == f(g(h(x))).

    Creates a single function that applies the given functions in reverse order.
    This follows mathematical function composition notation.

    Args:
        *funcs: Variable number of functions to compose. Each function should
            take a WaveformTrace and return a WaveformTrace.

    Returns:
        Composite function that applies all functions in reverse order.

    Raises:
        ValueError: If no functions provided.

    Example:
        >>> import oscura as osc
        >>> from functools import partial
        >>> # Create composed analysis function
        >>> analyze_signal = osc.compose(
        ...     osc.extract_thd,
        ...     partial(osc.fft, nfft=8192, window='hann'),
        ...     partial(osc.normalize, method='peak'),
        ...     partial(osc.low_pass, cutoff=5e6)
        ... )
        >>> # Apply to trace: low_pass -> normalize -> fft -> extract_thd
        >>> thd = analyze_signal(trace)

    References:
        API-002: Function Composition Operators
        toolz.functoolz
        https://github.com/pytoolz/toolz
    """
    if not funcs:
        raise ValueError("compose() requires at least one function")

    if len(funcs) == 1:
        return funcs[0]

    def composed(x: WaveformTrace) -> WaveformTrace:
        """Apply composed functions right-to-left."""
        # Apply functions in reverse order (right to left)
        return reduce(lambda val, func: func(val), reversed(funcs), x)

    # Preserve function metadata (handle functools.partial which lacks __name__)
    func_names = []
    for f in funcs:
        if hasattr(f, "__name__"):
            func_names.append(f.__name__)
        elif hasattr(f, "func"):  # functools.partial
            func_names.append(f.func.__name__)
        else:
            func_names.append(repr(f))
    composed.__name__ = "compose(" + ", ".join(func_names) + ")"
    composed.__doc__ = f"Composition of {len(funcs)} functions"

    return composed


def pipe(data: WaveformTrace, *funcs: TraceFunc) -> WaveformTrace:
    """Apply functions left-to-right: pipe(x, f, g, h) == h(g(f(x))).

    Applies the given functions sequentially to the data, passing the output
    of each function to the next. This is more intuitive for sequential
    processing pipelines.

    Args:
        data: Initial WaveformTrace to process.
        *funcs: Variable number of functions to apply sequentially.

    Returns:
        Transformed WaveformTrace after applying all functions.

    Example:
        >>> import oscura as osc
        >>> # Apply operations left-to-right
        >>> result = osc.pipe(
        ...     trace,
        ...     osc.low_pass(cutoff=1e6),
        ...     osc.resample(rate=1e9),
        ...     osc.fft(nfft=8192)
        ... )
        >>> # Equivalent to: fft(resample(low_pass(trace)))

    Advanced Example:
        >>> # Use with partial application
        >>> from functools import partial
        >>> result = osc.pipe(
        ...     trace,
        ...     partial(osc.low_pass, cutoff=1e6),
        ...     partial(osc.normalize, method='zscore'),
        ...     partial(osc.fft, nfft=8192, window='hann')
        ... )

    References:
        API-002: Function Composition Operators
        toolz.pipe
    """
    # Apply functions left-to-right
    return reduce(lambda val, func: func(val), funcs, data)


class Composable:
    """Mixin class to enable >> operator for function composition.

    This class provides the __rshift__ operator to enable pipe-style
    composition using the >> syntax. Intended to be mixed into WaveformTrace
    or used as a wrapper for transformer functions.

    Example:
        >>> # Enable >> operator on WaveformTrace
        >>> result = trace >> low_pass(1e6) >> normalize() >> fft()
        >>> # Equivalent to: fft(normalize(low_pass(trace)))

    References:
        API-002: Function Composition Operators
    """

    def __rshift__(self, func: Callable[[Any], Any]) -> Any:
        """Enable >> operator for function application.

        Args:
            func: Function to apply to self.

        Returns:
            Result of applying func to self.

        Example:
            >>> result = trace >> low_pass(1e6)
        """
        return func(self)


def make_composable(func: Callable[..., WaveformTrace]) -> Callable[..., TraceFunc]:
    """Decorator to make a function support partial application and composition.

    Wraps a function so it can be used in compose() and pipe() with
    partial argument application.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function that returns a partially applied function when
        called without a trace argument.

    Example:
        >>> @make_composable
        ... def scale(trace, factor=1.0):
        ...     return WaveformTrace(
        ...         data=trace.data * factor,
        ...         metadata=trace.metadata
        ...     )
        >>> # Use with partial application
        >>> double = scale(factor=2.0)
        >>> result = double(trace)
        >>> # Or in pipe
        >>> result = pipe(trace, scale(factor=2.0), scale(factor=0.5))

    References:
        API-002: Function Composition Operators
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> TraceFunc | WaveformTrace:
        # If first arg is a WaveformTrace, apply function immediately
        if args and isinstance(args[0], WaveformTrace):
            return func(*args, **kwargs)

        # Otherwise, return a partially applied function
        def partial_func(trace: WaveformTrace) -> WaveformTrace:
            return func(trace, *args, **kwargs)

        return partial_func

    return wrapper  # type: ignore[return-value]


def curry(func: Callable[..., WaveformTrace]) -> Callable[..., TraceFunc]:
    """Curry a function for easier composition.

    Transforms a multi-argument function into a series of single-argument
    functions. Useful for creating reusable transformation functions.

    Args:
        func: Function to curry.

    Returns:
        Curried version of the function.

    Example:
        >>> @curry
        ... def scale_and_offset(trace, scale, offset):
        ...     return WaveformTrace(
        ...         data=trace.data * scale + offset,
        ...         metadata=trace.metadata
        ...     )
        >>> # Create specialized functions
        >>> double_and_shift = scale_and_offset(scale=2.0, offset=1.0)
        >>> result = double_and_shift(trace)

    References:
        API-002: Function Composition Operators
        Functional programming currying
    """

    @wraps(func)
    def curried(*args: Any, **kwargs: Any) -> TraceFunc | WaveformTrace:
        # If we have a WaveformTrace as first arg, apply immediately
        if args and isinstance(args[0], WaveformTrace):
            return func(*args, **kwargs)

        # Return a function that waits for the trace
        def partial(*more_args: Any, **more_kwargs: Any) -> WaveformTrace:
            all_args = args + more_args
            all_kwargs = {**kwargs, **more_kwargs}
            return func(*all_args, **all_kwargs)

        return partial

    return curried  # type: ignore[return-value]


__all__ = [
    "Composable",
    "compose",
    "curry",
    "make_composable",
    "pipe",
]
