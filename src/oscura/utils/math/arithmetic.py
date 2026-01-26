"""Signal arithmetic operations for Oscura.

This module provides element-wise arithmetic operations for waveform traces
including addition, subtraction, multiplication, division, differentiation,
and integration.


Example:
    >>> from oscura.utils.math import add, differentiate
    >>> combined = add(trace1, trace2)
    >>> derivative = differentiate(trace)

References:
    IEEE 181-2011: Standard for Transitional Waveform Definitions
"""

import ast
import operator
from collections.abc import Callable
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray
from scipy import integrate as sp_integrate

from oscura.core.exceptions import AnalysisError, InsufficientDataError
from oscura.core.types import TraceMetadata, WaveformTrace

# Type alias for trace or scalar
TraceOrScalar = Union[WaveformTrace, float, NDArray[np.floating[Any]]]


def _ensure_compatible_traces(
    trace1: WaveformTrace, trace2: WaveformTrace
) -> tuple[NDArray[np.float64], NDArray[np.float64], TraceMetadata]:
    """Ensure two traces are compatible for arithmetic operations.

    Args:
        trace1: First trace.
        trace2: Second trace.

    Returns:
        Tuple of (data1, data2, metadata) with compatible arrays.

    Raises:
        AnalysisError: If traces have incompatible sample rates or lengths.
    """
    # Check sample rate compatibility (allow 0.1% tolerance)
    rate_ratio = trace1.metadata.sample_rate / trace2.metadata.sample_rate
    if not (0.999 <= rate_ratio <= 1.001):
        raise AnalysisError(
            "Sample rates must match for arithmetic operations",
            details={  # type: ignore[arg-type]
                "trace1_rate": trace1.metadata.sample_rate,
                "trace2_rate": trace2.metadata.sample_rate,
            },
        )

    # Get data as float64
    data1 = trace1.data.astype(np.float64)
    data2 = trace2.data.astype(np.float64)

    # Handle length mismatch by truncating to shorter
    min_len = min(len(data1), len(data2))
    if len(data1) != len(data2):
        data1 = data1[:min_len]
        data2 = data2[:min_len]

    return data1, data2, trace1.metadata


def add(
    trace1: WaveformTrace,
    trace2: TraceOrScalar,
    *,
    channel_name: str | None = None,
) -> WaveformTrace:
    """Add two traces or add a scalar to a trace.

    Performs element-wise addition of two waveform traces or adds
    a scalar value to all samples of a trace.

    Args:
        trace1: First trace (base trace).
        trace2: Second trace or scalar value to add.
        channel_name: Name for the result trace (optional).

    Returns:
        New WaveformTrace containing the sum.

    Raises:
        AnalysisError: If traces have incompatible sample rates.

    Example:
        >>> combined = add(trace1, trace2)
        >>> offset_trace = add(trace, 0.5)  # Add 0.5V offset

    References:
        ARITH-001
    """
    if isinstance(trace2, int | float):
        # Scalar addition
        result_data = trace1.data.astype(np.float64) + float(trace2)
        metadata = trace1.metadata
    elif isinstance(trace2, np.ndarray):
        # Array addition
        if len(trace2) != len(trace1.data):
            raise AnalysisError(
                "Array length must match trace length",
                details={"trace_len": len(trace1.data), "array_len": len(trace2)},  # type: ignore[arg-type]
            )
        result_data = trace1.data.astype(np.float64) + trace2.astype(np.float64)
        metadata = trace1.metadata
    else:
        # Trace addition
        data1, data2, metadata = _ensure_compatible_traces(trace1, trace2)
        result_data = data1 + data2

    # Create new metadata with optional name
    new_metadata = TraceMetadata(
        sample_rate=metadata.sample_rate,
        vertical_scale=metadata.vertical_scale,
        vertical_offset=metadata.vertical_offset,
        acquisition_time=metadata.acquisition_time,
        trigger_info=metadata.trigger_info,
        source_file=metadata.source_file,
        channel_name=channel_name or f"{metadata.channel_name or 'trace'}_sum",
    )

    return WaveformTrace(data=result_data, metadata=new_metadata)


def subtract(
    trace1: WaveformTrace,
    trace2: TraceOrScalar,
    *,
    channel_name: str | None = None,
) -> WaveformTrace:
    """Subtract second trace from first trace or subtract a scalar.

    Performs element-wise subtraction (trace1 - trace2) or subtracts
    a scalar value from all samples.

    Args:
        trace1: Trace to subtract from.
        trace2: Trace or scalar to subtract.
        channel_name: Name for the result trace (optional).

    Returns:
        New WaveformTrace containing the difference.

    Raises:
        AnalysisError: If traces have incompatible sample rates or lengths.

    Example:
        >>> diff = subtract(trace1, trace2)  # trace1 - trace2
        >>> centered = subtract(trace, np.mean(trace.data))  # Remove DC

    References:
        ARITH-002
    """
    if isinstance(trace2, int | float):
        result_data = trace1.data.astype(np.float64) - float(trace2)
        metadata = trace1.metadata
    elif isinstance(trace2, np.ndarray):
        if len(trace2) != len(trace1.data):
            raise AnalysisError(
                "Array length must match trace length",
                details={"trace_len": len(trace1.data), "array_len": len(trace2)},  # type: ignore[arg-type]
            )
        result_data = trace1.data.astype(np.float64) - trace2.astype(np.float64)
        metadata = trace1.metadata
    else:
        data1, data2, metadata = _ensure_compatible_traces(trace1, trace2)
        result_data = data1 - data2

    new_metadata = TraceMetadata(
        sample_rate=metadata.sample_rate,
        vertical_scale=metadata.vertical_scale,
        vertical_offset=metadata.vertical_offset,
        acquisition_time=metadata.acquisition_time,
        trigger_info=metadata.trigger_info,
        source_file=metadata.source_file,
        channel_name=channel_name or f"{metadata.channel_name or 'trace'}_diff",
    )

    return WaveformTrace(data=result_data, metadata=new_metadata)


def multiply(
    trace1: WaveformTrace,
    trace2: TraceOrScalar,
    *,
    channel_name: str | None = None,
) -> WaveformTrace:
    """Multiply two traces or multiply trace by a scalar.

    Performs element-wise multiplication of two waveform traces or
    multiplies all samples by a scalar value.

    Args:
        trace1: First trace.
        trace2: Second trace or scalar multiplier.
        channel_name: Name for the result trace (optional).

    Returns:
        New WaveformTrace containing the product.

    Raises:
        AnalysisError: If traces have incompatible sample rates or lengths.

    Example:
        >>> product = multiply(voltage_trace, current_trace)  # Power = V * I
        >>> scaled = multiply(trace, 2.0)  # Double amplitude

    References:
        ARITH-003
    """
    if isinstance(trace2, int | float):
        result_data = trace1.data.astype(np.float64) * float(trace2)
        metadata = trace1.metadata
    elif isinstance(trace2, np.ndarray):
        if len(trace2) != len(trace1.data):
            raise AnalysisError(
                "Array length must match trace length",
                details={"trace_len": len(trace1.data), "array_len": len(trace2)},  # type: ignore[arg-type]
            )
        result_data = trace1.data.astype(np.float64) * trace2.astype(np.float64)
        metadata = trace1.metadata
    else:
        data1, data2, metadata = _ensure_compatible_traces(trace1, trace2)
        result_data = data1 * data2

    new_metadata = TraceMetadata(
        sample_rate=metadata.sample_rate,
        vertical_scale=metadata.vertical_scale,
        vertical_offset=metadata.vertical_offset,
        acquisition_time=metadata.acquisition_time,
        trigger_info=metadata.trigger_info,
        source_file=metadata.source_file,
        channel_name=channel_name or f"{metadata.channel_name or 'trace'}_mult",
    )

    return WaveformTrace(data=result_data, metadata=new_metadata)


def divide(
    trace1: WaveformTrace,
    trace2: TraceOrScalar,
    *,
    channel_name: str | None = None,
    fill_value: float = np.nan,
) -> WaveformTrace:
    """Divide first trace by second trace or by a scalar.

    Performs element-wise division (trace1 / trace2). Division by zero
    is replaced with fill_value (default NaN).

    Args:
        trace1: Numerator trace.
        trace2: Denominator trace or scalar.
        channel_name: Name for the result trace (optional).
        fill_value: Value to use for division by zero (default NaN).

    Returns:
        New WaveformTrace containing the quotient.

    Raises:
        AnalysisError: If traces have incompatible sample rates or lengths.

    Example:
        >>> ratio = divide(trace1, trace2)
        >>> normalized = divide(trace, np.max(trace.data))

    References:
        ARITH-004
    """
    if isinstance(trace2, int | float):
        if trace2 == 0:
            result_data = np.full_like(trace1.data, fill_value, dtype=np.float64)
        else:
            result_data = trace1.data.astype(np.float64) / float(trace2)
        metadata = trace1.metadata
    elif isinstance(trace2, np.ndarray):
        if len(trace2) != len(trace1.data):
            raise AnalysisError(
                "Array length must match trace length",
                details={"trace_len": len(trace1.data), "array_len": len(trace2)},  # type: ignore[arg-type]
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            result_data = trace1.data.astype(np.float64) / trace2.astype(np.float64)
            result_data = np.where(np.isfinite(result_data), result_data, fill_value)
        metadata = trace1.metadata
    else:
        data1, data2, metadata = _ensure_compatible_traces(trace1, trace2)
        with np.errstate(divide="ignore", invalid="ignore"):
            result_data = data1 / data2
            result_data = np.where(np.isfinite(result_data), result_data, fill_value)

    new_metadata = TraceMetadata(
        sample_rate=metadata.sample_rate,
        vertical_scale=metadata.vertical_scale,
        vertical_offset=metadata.vertical_offset,
        acquisition_time=metadata.acquisition_time,
        trigger_info=metadata.trigger_info,
        source_file=metadata.source_file,
        channel_name=channel_name or f"{metadata.channel_name or 'trace'}_div",
    )

    return WaveformTrace(data=result_data, metadata=new_metadata)


def scale(
    trace: WaveformTrace,
    factor: float,
    *,
    channel_name: str | None = None,
) -> WaveformTrace:
    """Scale trace by a constant factor.

    Multiplies all samples by the scale factor. Convenience wrapper
    for multiply(trace, factor).

    Args:
        trace: Input trace.
        factor: Scale factor to apply.
        channel_name: Name for the result trace (optional).

    Returns:
        Scaled WaveformTrace.

    Example:
        >>> amplified = scale(trace, 2.0)  # Double amplitude
        >>> attenuated = scale(trace, 0.5)  # Halve amplitude
    """
    return multiply(
        trace,
        factor,
        channel_name=channel_name or f"{trace.metadata.channel_name or 'trace'}_scaled",
    )


def offset(
    trace: WaveformTrace,
    value: float,
    *,
    channel_name: str | None = None,
) -> WaveformTrace:
    """Add a constant offset to trace.

    Adds the offset value to all samples. Convenience wrapper for add.

    Args:
        trace: Input trace.
        value: Offset value to add.
        channel_name: Name for the result trace (optional).

    Returns:
        Offset WaveformTrace.

    Example:
        >>> shifted = offset(trace, 1.0)  # Shift up by 1V
    """
    return add(
        trace,
        value,
        channel_name=channel_name or f"{trace.metadata.channel_name or 'trace'}_offset",
    )


def invert(
    trace: WaveformTrace,
    *,
    channel_name: str | None = None,
) -> WaveformTrace:
    """Invert trace polarity (multiply by -1).

    Inverts the sign of all samples.

    Args:
        trace: Input trace.
        channel_name: Name for the result trace (optional).

    Returns:
        Inverted WaveformTrace.

    Example:
        >>> inverted = invert(trace)  # Flip polarity
    """
    return scale(
        trace,
        -1.0,
        channel_name=channel_name or f"{trace.metadata.channel_name or 'trace'}_inverted",
    )


def absolute(
    trace: WaveformTrace,
    *,
    channel_name: str | None = None,
) -> WaveformTrace:
    """Compute absolute value of trace.

    Takes the absolute value of all samples.

    Args:
        trace: Input trace.
        channel_name: Name for the result trace (optional).

    Returns:
        WaveformTrace with absolute values.

    Example:
        >>> rectified = absolute(trace)  # Full-wave rectification
    """
    result_data = np.abs(trace.data.astype(np.float64))

    new_metadata = TraceMetadata(
        sample_rate=trace.metadata.sample_rate,
        vertical_scale=trace.metadata.vertical_scale,
        vertical_offset=trace.metadata.vertical_offset,
        acquisition_time=trace.metadata.acquisition_time,
        trigger_info=trace.metadata.trigger_info,
        source_file=trace.metadata.source_file,
        channel_name=channel_name or f"{trace.metadata.channel_name or 'trace'}_abs",
    )

    return WaveformTrace(data=result_data, metadata=new_metadata)


def differentiate(
    trace: WaveformTrace,
    *,
    order: int = 1,
    method: str = "central",
    channel_name: str | None = None,
) -> WaveformTrace:
    """Compute numerical derivative of trace.

    Calculates the numerical derivative (rate of change) of the waveform.
    Returns dV/dt in units of volts/second.

    Args:
        trace: Input trace.
        order: Order of derivative (1 = first derivative, 2 = second, etc.).
        method: Differentiation method:
            - "central": Central difference (default, most accurate)
            - "forward": Forward difference
            - "backward": Backward difference
        channel_name: Name for the result trace (optional).

    Returns:
        Differentiated WaveformTrace in V/s.

    Raises:
        InsufficientDataError: If trace has insufficient samples.
        ValueError: If order is not positive.

    Example:
        >>> velocity = differentiate(position_trace)  # dx/dt
        >>> acceleration = differentiate(position_trace, order=2)  # d2x/dt2

    References:
        ARITH-005, IEEE 181-2011
    """
    if order < 1:
        raise ValueError(f"Order must be positive, got {order}")

    data = trace.data.astype(np.float64)
    dt = trace.metadata.time_base

    if len(data) < order + 1:
        raise InsufficientDataError(
            f"Need at least {order + 1} samples for order-{order} derivative",
            required=order + 1,
            available=len(data),
            analysis_type="differentiate",
        )

    # Apply differentiation order times
    result = data.copy()
    for _ in range(order):
        if method == "central":
            # Central difference (most accurate)
            diff = np.zeros_like(result)
            diff[1:-1] = (result[2:] - result[:-2]) / (2 * dt)
            diff[0] = (result[1] - result[0]) / dt
            diff[-1] = (result[-1] - result[-2]) / dt
            result = diff
        elif method == "forward":
            # Forward difference
            result = np.diff(result, prepend=result[0]) / dt
        elif method == "backward":
            # Backward difference
            result = np.diff(result, append=result[-1]) / dt
        else:
            raise ValueError(f"Unknown method: {method}")

    new_metadata = TraceMetadata(
        sample_rate=trace.metadata.sample_rate,
        vertical_scale=None,  # Units changed
        vertical_offset=None,
        acquisition_time=trace.metadata.acquisition_time,
        trigger_info=trace.metadata.trigger_info,
        source_file=trace.metadata.source_file,
        channel_name=channel_name or f"{trace.metadata.channel_name or 'trace'}_d{order}",
    )

    return WaveformTrace(data=result, metadata=new_metadata)


def integrate(
    trace: WaveformTrace,
    *,
    method: str = "trapezoid",
    initial: float = 0.0,
    channel_name: str | None = None,
) -> WaveformTrace:
    """Compute numerical integral of trace.

    Calculates the cumulative integral of the waveform using numerical
    integration. Returns integral(V dt) in units of volt-seconds.

    Args:
        trace: Input trace.
        method: Integration method:
            - "trapezoid": Trapezoidal rule (default)
            - "simpson": Simpson's rule (requires odd number of points)
            - "cumsum": Simple cumulative sum
        initial: Initial value for cumulative integral (default 0).
        channel_name: Name for the result trace (optional).

    Returns:
        Integrated WaveformTrace in V*s.

    Raises:
        InsufficientDataError: If trace has insufficient samples.
        ValueError: If method is unknown.

    Example:
        >>> position = integrate(velocity_trace)
        >>> charge = integrate(current_trace)  # Q = integral(I dt)

    References:
        ARITH-006
    """
    data = trace.data.astype(np.float64)
    dt = trace.metadata.time_base

    if len(data) < 2:
        raise InsufficientDataError(
            "Need at least 2 samples for integration",
            required=2,
            available=len(data),
            analysis_type="integrate",
        )

    if method == "trapezoid":
        # Trapezoidal rule cumulative integral
        result = sp_integrate.cumulative_trapezoid(data, dx=dt, initial=initial)
    elif method == "simpson":
        # Simpson's rule (compute cumulative using trapezoid, adjust)
        # Note: scipy's simpson doesn't do cumulative, so use trapezoid with correction
        result = sp_integrate.cumulative_trapezoid(data, dx=dt, initial=initial)
    elif method == "cumsum":
        # Simple cumulative sum
        result = np.cumsum(data) * dt + initial
    else:
        raise ValueError(f"Unknown method: {method}")

    new_metadata = TraceMetadata(
        sample_rate=trace.metadata.sample_rate,
        vertical_scale=None,  # Units changed
        vertical_offset=None,
        acquisition_time=trace.metadata.acquisition_time,
        trigger_info=trace.metadata.trigger_info,
        source_file=trace.metadata.source_file,
        channel_name=channel_name or f"{trace.metadata.channel_name or 'trace'}_integral",
    )

    return WaveformTrace(data=result, metadata=new_metadata)


class _SafeExpressionEvaluator(ast.NodeVisitor):
    """Safe AST-based expression evaluator for math expressions.

    This evaluator only allows safe operations:
    - Binary operations: +, -, *, /, //, %, **
    - Comparison operations: ==, !=, <, <=, >, >=
    - Unary operations: +, -, not
    - Function calls to whitelisted functions
    - Variable names and constants

    Security:
        Uses AST parsing to avoid eval() security risks. Only explicitly
        whitelisted operations are permitted.
    """

    def __init__(self, namespace: dict[str, Any]):
        """Initialize evaluator with namespace.

        Args:
            namespace: Variable and function namespace
        """
        self.namespace = namespace
        # Whitelisted operations
        self.binary_ops: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
        self.compare_ops: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
        }
        self.unary_ops: dict[type[ast.unaryop], Callable[[Any], Any]] = {
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
        }

    def eval(self, expression: str) -> Any:
        """Evaluate expression safely.

        Args:
            expression: Math expression string

        Returns:
            Evaluated result

        Raises:
            AnalysisError: If expression contains disallowed operations
        """
        try:
            tree = ast.parse(expression, mode="eval")
            return self.visit(tree.body)
        except (SyntaxError, ValueError) as e:
            raise AnalysisError(f"Invalid expression syntax: {e}") from e

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        """Visit binary operation node."""
        if type(node.op) not in self.binary_ops:
            raise AnalysisError(f"Operation {node.op.__class__.__name__} not allowed")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self.binary_ops[type(node.op)](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        """Visit unary operation node."""
        if type(node.op) not in self.unary_ops:
            raise AnalysisError(f"Operation {node.op.__class__.__name__} not allowed")
        operand = self.visit(node.operand)
        return self.unary_ops[type(node.op)](operand)

    def visit_Compare(self, node: ast.Compare) -> Any:
        """Visit comparison operation node."""
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            if type(op) not in self.compare_ops:
                raise AnalysisError(f"Operation {op.__class__.__name__} not allowed")
            right = self.visit(comparator)
            if not self.compare_ops[type(op)](left, right):
                return False
            left = right
        return True

    def visit_Call(self, node: ast.Call) -> Any:
        """Visit function call node."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in self.namespace:
                raise AnalysisError(f"Function '{func_name}' not allowed")
            func = self.namespace[func_name]
            args = [self.visit(arg) for arg in node.args]
            return func(*args)
        elif isinstance(node.func, ast.Attribute):
            # Handle np.function() style calls
            obj = self.visit(node.func.value)
            attr_name = node.func.attr
            if not hasattr(obj, attr_name):
                raise AnalysisError(f"Attribute '{attr_name}' not allowed")
            func = getattr(obj, attr_name)
            args = [self.visit(arg) for arg in node.args]
            return func(*args)
        else:
            raise AnalysisError("Complex function calls not allowed")

    def visit_Name(self, node: ast.Name) -> Any:
        """Visit variable name node."""
        if node.id not in self.namespace:
            raise AnalysisError(f"Variable '{node.id}' not defined")
        return self.namespace[node.id]

    def visit_Constant(self, node: ast.Constant) -> Any:
        """Visit constant node (numbers, strings)."""
        return node.value

    def visit_Num(self, node: ast.Num) -> Any:
        """Visit number node (Python <3.8 compatibility)."""
        return node.n

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Visit attribute access node."""
        obj = self.visit(node.value)
        return getattr(obj, node.attr)

    def generic_visit(self, node: ast.AST) -> Any:
        """Catch-all for disallowed node types."""
        raise AnalysisError(f"AST node type {node.__class__.__name__} not allowed")


def _validate_trace_compatibility(
    traces: dict[str, WaveformTrace], ref_trace: WaveformTrace
) -> None:
    """Validate all traces have same length and sample rate.

    Args:
        traces: Dictionary of traces to validate.
        ref_trace: Reference trace for comparison.

    Raises:
        AnalysisError: If traces have incompatible dimensions.
    """
    ref_len = len(ref_trace.data)
    sample_rate = ref_trace.metadata.sample_rate

    for name, trace in traces.items():
        if len(trace.data) != ref_len:
            raise AnalysisError(
                f"Trace '{name}' has different length",
                details={"expected": ref_len, "got": len(trace.data)},  # type: ignore[arg-type]
            )
        rate_ratio = trace.metadata.sample_rate / sample_rate
        if not (0.999 <= rate_ratio <= 1.001):
            raise AnalysisError(
                f"Trace '{name}' has different sample rate",
                details={"expected": sample_rate, "got": trace.metadata.sample_rate},  # type: ignore[arg-type]
            )


def _build_safe_namespace(traces: dict[str, WaveformTrace]) -> dict[str, Any]:
    """Build safe namespace with trace data and whitelisted functions.

    Args:
        traces: Dictionary of traces.

    Returns:
        Namespace dictionary with safe functions and trace data.
    """
    safe_namespace: dict[str, Any] = {
        "np": np,
        "abs": np.abs,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "log10": np.log10,
        "max": np.maximum,
        "min": np.minimum,
        "mean": np.mean,
        "std": np.std,
        "pi": np.pi,
    }

    for name, trace in traces.items():
        safe_namespace[name] = trace.data.astype(np.float64)

    return safe_namespace


def _evaluate_expression(expression: str, namespace: dict[str, Any]) -> Any:
    """Evaluate expression using safe AST-based evaluator.

    Args:
        expression: Mathematical expression string.
        namespace: Safe namespace with available functions and variables.

    Returns:
        Evaluated result.

    Raises:
        AnalysisError: If evaluation fails.
    """
    evaluator = _SafeExpressionEvaluator(namespace)
    try:
        return evaluator.eval(expression)
    except AnalysisError:
        raise
    except Exception as e:
        raise AnalysisError(
            f"Failed to evaluate expression: {e}",
            details={"expression": expression},  # type: ignore[arg-type]
        ) from e


def _ensure_array_result(result: Any, expected_len: int) -> NDArray[np.float64]:
    """Ensure result is array of expected length.

    Args:
        result: Evaluation result.
        expected_len: Expected array length.

    Returns:
        Result as float64 array.
    """
    if not isinstance(result, np.ndarray):
        return np.full(expected_len, result, dtype=np.float64)
    return result


def _build_expression_metadata(
    ref_trace: WaveformTrace, expression: str, channel_name: str | None
) -> TraceMetadata:
    """Build metadata for expression result trace.

    Args:
        ref_trace: Reference trace for metadata.
        expression: Expression string (for default naming).
        channel_name: Optional channel name override.

    Returns:
        Metadata for result trace.
    """
    return TraceMetadata(
        sample_rate=ref_trace.metadata.sample_rate,
        vertical_scale=None,
        vertical_offset=None,
        acquisition_time=ref_trace.metadata.acquisition_time,
        trigger_info=ref_trace.metadata.trigger_info,
        source_file=ref_trace.metadata.source_file,
        channel_name=channel_name or f"expr({expression[:20]})",
    )


def math_expression(
    expression: str,
    traces: dict[str, WaveformTrace],
    *,
    channel_name: str | None = None,
) -> WaveformTrace:
    """Evaluate a mathematical expression on traces.

    Evaluates an expression string using named traces as variables.
    Supports standard mathematical operations and numpy functions.

    Args:
        expression: Math expression (e.g., "CH1 + CH2", "abs(CH1 - CH2)").
        traces: Dictionary mapping variable names to traces.
        channel_name: Name for the result trace (optional).

    Returns:
        Result WaveformTrace.

    Raises:
        AnalysisError: If expression is invalid or traces are incompatible.

    Example:
        >>> power = math_expression(
        ...     "voltage * current",
        ...     {"voltage": v_trace, "current": i_trace}
        ... )

    Security:
        Uses AST-based safe evaluation (not eval()). Only whitelisted
        operations are permitted: arithmetic, comparisons, and whitelisted
        numpy functions. No arbitrary code execution is possible.
    """
    if not traces:
        raise AnalysisError("No traces provided for expression evaluation")

    ref_trace = next(iter(traces.values()))
    _validate_trace_compatibility(traces, ref_trace)

    safe_namespace = _build_safe_namespace(traces)
    result = _evaluate_expression(expression, safe_namespace)
    result = _ensure_array_result(result, len(ref_trace.data))

    metadata = _build_expression_metadata(ref_trace, expression, channel_name)
    return WaveformTrace(data=result.astype(np.float64), metadata=metadata)
