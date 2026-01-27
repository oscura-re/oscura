"""Oscura DSL - Domain-Specific Language for trace analysis.

Provides a simple, declarative language for defining trace analysis workflows.

Example usage:
    ```python
    from oscura.api.dsl import execute_dsl

    # Execute DSL script
    script = '''
    $data = load "capture.csv"
    $filtered = $data | filter lowpass 1000
    $rise = $filtered | measure rise_time
    '''

    env = execute_dsl(script)
    print(f"Rise time: {env['$rise']}")
    ```

    Or start interactive REPL:
    ```python
    from oscura.api.dsl import start_repl
    start_repl()
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from oscura.api.dsl.commands import BUILTIN_COMMANDS
from oscura.api.dsl.interpreter import Interpreter, InterpreterError, execute_dsl
from oscura.api.dsl.parser import (
    Assignment,
    Command,
    Expression,
    ForLoop,
    FunctionCall,
    Lexer,
    Literal,
    Parser,
    Pipeline,
    Statement,
    Token,
    TokenType,
    Variable,
    parse_dsl,
)
from oscura.api.dsl.repl import REPL, start_repl


# Legacy API dataclass for backwards compatibility
@dataclass
class DSLExpression:
    """Legacy DSL expression dataclass for backwards compatibility.

    Represents a single DSL operation with arguments and optional chaining.

    Attributes:
        operation: Operation name (e.g., "fft", "lowpass", "mean").
        args: Positional arguments.
        kwargs: Keyword arguments.
        chain: Optional chained expression to execute after this one.
    """

    operation: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    chain: DSLExpression | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert expression to dictionary representation.

        Returns:
            Dictionary with operation, args, kwargs, and optionally chain.
        """
        result: dict[str, Any] = {
            "operation": self.operation,
            "args": self.args,
            "kwargs": self.kwargs,
        }
        if self.chain is not None:
            result["chain"] = self.chain.to_dict()
        return result


# Legacy API parser class
class DSLParser:
    """Legacy parser class for backwards compatibility.

    Parses DSL expression strings into DSLExpression objects.
    """

    def __init__(self) -> None:
        """Initialize parser state."""
        self._pos = 0
        self._text = ""

    def parse(self, text: str) -> DSLExpression:
        """Parse DSL expression string.

        Args:
            text: Expression string to parse.

        Returns:
            Parsed DSLExpression.

        Raises:
            ValueError: If expression is invalid.
        """
        self._text = text.strip()
        self._pos = 0

        if not self._text:
            raise ValueError("Expected identifier")

        return self._parse_chain()

    def _parse_chain(self) -> DSLExpression:
        """Parse potentially chained expressions."""
        expr = self._parse_operation()

        # Check for pipe operator
        self._skip_whitespace()
        if self._peek() == "|":
            self._advance()  # consume |
            chain = self._parse_chain()
            expr.chain = chain

        return expr

    def _parse_operation(self) -> DSLExpression:
        """Parse a single operation."""
        self._skip_whitespace()

        # Parse operation name
        operation = self._parse_identifier()
        if not operation:
            raise ValueError("Expected identifier")

        # Validate operation
        valid_ops = {
            "mean",
            "std",
            "min",
            "max",
            "rms",
            "fft",
            "psd",
            "lowpass",
            "highpass",
            "bandpass",
            "normalize",
            "slice",
            "resample",
            "load",
            "filter",
        }
        if operation not in valid_ops:
            raise ValueError(f"Unknown operation: {operation}")

        # Parse arguments if present
        args: list[Any] = []
        kwargs: dict[str, Any] = {}

        self._skip_whitespace()
        if self._peek() == "(":
            self._advance()  # consume (
            args, kwargs = self._parse_args()
            if self._peek() != ")":
                raise ValueError("Expected ')'")
            self._advance()  # consume )

        return DSLExpression(operation=operation, args=args, kwargs=kwargs)

    def _parse_args(self) -> tuple[list[Any], dict[str, Any]]:
        """Parse function arguments."""
        args: list[Any] = []
        kwargs: dict[str, Any] = {}

        self._skip_whitespace()
        if self._peek() == ")":
            return args, kwargs

        while True:
            self._skip_whitespace()

            # Try to parse keyword argument (identifier=value)
            saved_pos = self._pos
            identifier = self._parse_identifier()
            self._skip_whitespace()

            if identifier and self._peek() == "=":
                # It's a keyword argument
                self._advance()  # consume =
                value = self._parse_value()
                kwargs[identifier] = value
            else:
                # It's a positional argument - restore position and parse value
                self._pos = saved_pos
                value = self._parse_value()
                args.append(value)

            self._skip_whitespace()
            if self._peek() == ",":
                self._advance()
                self._skip_whitespace()
                # Check for trailing comma before closing paren
                if self._peek() == ")":
                    break
                continue
            break

        return args, kwargs

    def _parse_value(self) -> Any:
        """Parse a value (number, string, list, bool, etc)."""
        self._skip_whitespace()

        ch = self._peek()

        # String
        if ch in ('"', "'"):
            return self._parse_string()

        # List
        if ch == "[":
            return self._parse_list()

        # Number or identifier
        if ch.isdigit() or ch in ("-", "."):
            return self._parse_number()

        # Boolean/None/identifier
        identifier = self._parse_identifier()
        if identifier == "True":
            return True
        elif identifier == "False":
            return False
        elif identifier == "None":
            return None
        elif identifier:
            return identifier
        else:
            raise ValueError(f"Unexpected character: {ch}")

    def _parse_number(self) -> int | float:
        """Parse number (int or float)."""
        start = self._pos
        has_dot = False
        has_e = False

        # Handle negative sign
        if self._peek() == "-":
            self._advance()

        # Leading dot
        if self._peek() == ".":
            has_dot = True
            self._advance()

        while self._pos < len(self._text):
            ch = self._peek()
            if ch.isdigit():
                self._advance()
            elif ch == "." and not has_dot and not has_e:
                has_dot = True
                self._advance()
            elif ch in ("e", "E") and not has_e:
                has_e = True
                self._advance()
                # Handle +/- after e
                if self._peek() in ("+", "-"):
                    self._advance()
            else:
                break

        num_str = self._text[start : self._pos]
        if "." in num_str or "e" in num_str or "E" in num_str:
            return float(num_str)
        else:
            return int(num_str)

    def _parse_string(self) -> str:
        """Parse string literal."""
        quote = self._advance()  # get quote character
        start = self._pos
        while self._pos < len(self._text):
            ch = self._peek()
            if ch == quote:
                result = self._text[start : self._pos]
                self._advance()  # consume closing quote
                return result
            elif ch == "\\":
                # Skip escaped character
                self._advance()
                if self._pos < len(self._text):
                    self._advance()
            else:
                self._advance()

        raise ValueError("Unterminated string")

    def _parse_list(self) -> list[Any]:
        """Parse list literal."""
        self._advance()  # consume [
        items: list[Any] = []

        self._skip_whitespace()
        if self._peek() == "]":
            self._advance()
            return items

        while True:
            items.append(self._parse_value())
            self._skip_whitespace()
            if self._peek() == ",":
                self._advance()
                self._skip_whitespace()
                # Handle trailing comma
                if self._peek() == "]":
                    break
                continue
            break

        if self._peek() != "]":
            raise ValueError("Unterminated list")
        self._advance()

        return items

    def _parse_identifier(self) -> str:
        """Parse identifier."""
        start = self._pos
        while self._pos < len(self._text):
            ch = self._peek()
            if ch.isalnum() or ch == "_":
                self._advance()
            else:
                break
        return self._text[start : self._pos]

    def _skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self._pos < len(self._text) and self._text[self._pos].isspace():
            self._pos += 1

    def _peek(self) -> str:
        """Peek at current character."""
        if self._pos < len(self._text):
            return self._text[self._pos]
        return ""

    def _advance(self) -> str:
        """Advance and return current character."""
        if self._pos < len(self._text):
            ch = self._text[self._pos]
            self._pos += 1
            return ch
        return ""


def parse_expression(text: str) -> DSLExpression:
    """Parse DSL expression string (legacy API).

    Args:
        text: Expression string to parse.

    Returns:
        Parsed DSLExpression.

    Example:
        >>> expr = parse_expression("lowpass(cutoff=1e6)")
        >>> print(expr.operation)
        lowpass
    """
    parser = DSLParser()
    return parser.parse(text)


def analyze(data: Any, expression_str: str) -> Any:
    """Execute DSL expression on data (legacy API).

    Args:
        data: NumPy array to analyze.
        expression_str: DSL expression string.

    Returns:
        Analysis result.

    Example:
        >>> import numpy as np
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> analyze(data, "mean")
        3.0
    """
    import numpy as np

    # Parse expression using legacy parser
    expr = parse_expression(expression_str)

    # Execute using legacy executor
    executor = DSLExecutor()
    return executor.execute(expr, data)


# DSLExecutor class for backwards compatibility
class DSLExecutor:
    """Legacy executor class for backwards compatibility.

    Executes DSL operations on numpy arrays for testing purposes.
    """

    def __init__(self) -> None:
        """Initialize executor with operation registry."""
        self._operations: dict[str, Any] = {
            "mean": lambda data, **kw: float(__import__("numpy").mean(data)),
            "std": lambda data, **kw: float(__import__("numpy").std(data)),
            "min": lambda data, **kw: float(__import__("numpy").min(data)),
            "max": lambda data, **kw: float(__import__("numpy").max(data)),
            "rms": lambda data, **kw: float(
                __import__("numpy").sqrt(__import__("numpy").mean(data**2))
            ),
            "fft": lambda data, **kw: __import__("numpy").fft.fft(
                data, n=kw.get("nfft", len(data))
            ),
            "normalize": self._normalize,
            "lowpass": self._lowpass,
            "highpass": self._highpass,
            "bandpass": self._bandpass,
            "slice": self._slice,
            "resample": self._resample,
            "psd": self._psd,
        }

    def execute(self, expr: DSLExpression, data: Any) -> Any:
        """Execute DSL expression on data.

        Args:
            expr: DSLExpression to execute.
            data: NumPy array to process.

        Returns:
            Execution result.

        Raises:
            ValueError: If operation is unknown.
        """
        import numpy as np

        # Execute main operation
        if expr.operation not in self._operations:
            raise ValueError(f"Unknown operation: {expr.operation}")

        # Call operation with positional args and keyword args
        # Convert positional args to operation-specific kwargs for known operations
        kwargs = dict(expr.kwargs)

        # Handle slice operation with positional args: slice(start, end)
        if expr.operation == "slice" and expr.args:
            if len(expr.args) >= 1 and "start" not in kwargs:
                kwargs["start"] = expr.args[0]
            if len(expr.args) >= 2 and "end" not in kwargs:
                kwargs["end"] = expr.args[1]

        result = self._operations[expr.operation](data, **kwargs)

        # Execute chain if present
        if expr.chain is not None:
            if not isinstance(result, np.ndarray):
                raise ValueError(f"Cannot chain after {expr.operation}")
            result = self.execute(expr.chain, result)

        return result

    def _normalize(self, data: Any, method: str = "minmax", **kwargs: Any) -> Any:
        """Normalize data."""
        import numpy as np

        if method == "minmax":
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max == data_min:
                return data
            return (data - data_min) / (data_max - data_min)
        elif method == "zscore":
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return data - mean
            return (data - mean) / std
        else:
            return data

    def _lowpass(self, data: Any, cutoff: float = 1e5, fs: float = 1e6, **kwargs: Any) -> Any:
        """Apply lowpass filter."""
        from scipy.signal import butter, filtfilt

        nyquist = fs / 2
        normalized_cutoff = cutoff / nyquist
        b, a = butter(4, normalized_cutoff, btype="low")
        return filtfilt(b, a, data)

    def _highpass(self, data: Any, cutoff: float = 1e5, fs: float = 1e6, **kwargs: Any) -> Any:
        """Apply highpass filter."""
        from scipy.signal import butter, filtfilt

        nyquist = fs / 2
        normalized_cutoff = cutoff / nyquist
        b, a = butter(4, normalized_cutoff, btype="high")
        return filtfilt(b, a, data)

    def _bandpass(
        self,
        data: Any,
        low: float = 1e3,
        high: float = 1e6,
        fs: float = 1e6,
        **kwargs: Any,
    ) -> Any:
        """Apply bandpass filter."""
        from scipy.signal import butter, filtfilt

        nyquist = fs / 2
        normalized_low = low / nyquist
        normalized_high = high / nyquist
        b, a = butter(4, [normalized_low, normalized_high], btype="band")
        return filtfilt(b, a, data)

    def _slice(self, data: Any, start: int = 0, end: int | None = None, **kwargs: Any) -> Any:
        """Slice data."""
        return data[start:end]

    def _resample(self, data: Any, factor: int = 2, **kwargs: Any) -> Any:
        """Resample data by decimation factor."""
        return data[::factor]

    def _psd(self, data: Any, nperseg: int = 256, **kwargs: Any) -> Any:
        """Compute power spectral density."""
        from scipy.signal import welch

        _, psd = welch(data, nperseg=nperseg)
        return psd


__all__ = [
    # Commands
    "BUILTIN_COMMANDS",
    # REPL
    "REPL",
    # AST nodes
    "Assignment",
    "Command",
    # Legacy API (backwards compatibility)
    "DSLExecutor",
    "DSLExpression",
    "DSLParser",
    "Expression",
    "ForLoop",
    "FunctionCall",
    # Interpreter
    "Interpreter",
    "InterpreterError",
    # Parser
    "Lexer",
    "Literal",
    "Parser",
    "Pipeline",
    "Statement",
    "Token",
    "TokenType",
    "Variable",
    "analyze",
    "execute_dsl",
    "parse_dsl",
    "parse_expression",
    "start_repl",
]
