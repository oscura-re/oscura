"""Domain-Specific Language (DSL) for signal analysis.

This module provides a simple DSL for expressing signal analysis
operations in a readable, declarative format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

__all__ = [
    "DSLExpression",
    "DSLParser",
    "analyze",
    "parse_expression",
]


@dataclass
class DSLExpression:
    """Parsed DSL expression.

    Attributes:
        operation: Operation name
        args: Positional arguments
        kwargs: Keyword arguments
        chain: Chained operation (if any)

    Example:
        >>> expr = DSLExpression(
        ...     operation="fft",
        ...     kwargs={"nfft": 8192}
        ... )

    References:
        API-010: Domain-Specific Language (DSL)
    """

    operation: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    chain: DSLExpression | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {"operation": self.operation, "args": self.args, "kwargs": self.kwargs}
        if self.chain:
            result["chain"] = self.chain.to_dict()
        return result


class DSLParser:
    """Parser for signal analysis DSL.

    Grammar:
        expression := operation | operation '|' expression
        operation := name | name '(' arguments ')'
        arguments := arg | arg ',' arguments
        arg := value | name '=' value
        value := number | string | list

    Example:
        >>> parser = DSLParser()
        >>> expr = parser.parse("lowpass(cutoff=1e6) | fft(nfft=8192)")
        >>> print(expr.operation)
        'lowpass'

    References:
        API-010: Domain-Specific Language (DSL)
    """

    OPERATIONS = {
        "load",
        "save",
        "export",
        "lowpass",
        "highpass",
        "bandpass",
        "notch",
        "filter",
        "fft",
        "ifft",
        "psd",
        "spectrogram",
        "cwt",
        "mean",
        "std",
        "min",
        "max",
        "rms",
        "peak_to_peak",
        "rise_time",
        "fall_time",
        "frequency",
        "period",
        "threshold",
        "edges",
        "pulses",
        "decode",
        "uart",
        "spi",
        "i2c",
        "can",
        "plot",
        "show",
        "histogram",
        "resample",
        "decimate",
        "interpolate",
        "normalize",
        "zscore",
        "scale",
        "clip",
        "slice",
        "select",
    }

    def __init__(self) -> None:
        """Initialize parser."""
        self._pos = 0
        self._text = ""

    def parse(self, text: str) -> DSLExpression:
        """Parse DSL expression.

        Args:
            text: DSL expression text

        Returns:
            Parsed expression
        """
        self._text = text.strip()
        self._pos = 0
        return self._parse_chain()

    def _parse_chain(self) -> DSLExpression:
        """Parse expression chain."""
        expr = self._parse_operation()

        self._skip_whitespace()
        if self._pos < len(self._text) and self._text[self._pos] == "|":
            self._pos += 1
            self._skip_whitespace()
            expr.chain = self._parse_chain()

        return expr

    def _parse_operation(self) -> DSLExpression:
        """Parse single operation."""
        self._skip_whitespace()

        # Parse operation name
        name = self._parse_identifier()
        if name not in self.OPERATIONS:
            raise ValueError(f"Unknown operation: {name}")

        self._skip_whitespace()

        # Check for arguments
        args: list[Any] = []
        kwargs: dict[str, Any] = {}

        if self._pos < len(self._text) and self._text[self._pos] == "(":
            self._pos += 1  # Skip '('
            args, kwargs = self._parse_arguments()

            if self._pos >= len(self._text) or self._text[self._pos] != ")":
                raise ValueError("Expected ')'")
            self._pos += 1  # Skip ')'

        return DSLExpression(operation=name, args=args, kwargs=kwargs)

    def _parse_arguments(self) -> tuple[list[Any], dict[str, Any]]:
        """Parse argument list."""
        args = []
        kwargs = {}

        while True:
            self._skip_whitespace()

            if self._pos >= len(self._text) or self._text[self._pos] == ")":
                break

            # Check for keyword argument
            start = self._pos
            name = self._try_parse_identifier()

            self._skip_whitespace()
            if name and self._pos < len(self._text) and self._text[self._pos] == "=":
                self._pos += 1  # Skip '='
                self._skip_whitespace()
                value = self._parse_value()
                kwargs[name] = value
            else:
                # Positional argument
                self._pos = start
                value = self._parse_value()
                args.append(value)

            self._skip_whitespace()
            if self._pos < len(self._text) and self._text[self._pos] == ",":
                self._pos += 1

        return args, kwargs

    def _parse_value(self) -> Any:
        """Parse a value."""
        self._skip_whitespace()

        if self._pos >= len(self._text):
            raise ValueError("Unexpected end of expression")

        char = self._text[self._pos]

        # String
        if char in "\"'":
            return self._parse_string()

        # List
        if char == "[":
            return self._parse_list()

        # Number or identifier
        return self._parse_number_or_identifier()

    def _parse_string(self) -> str:
        """Parse string literal."""
        quote = self._text[self._pos]
        self._pos += 1
        start = self._pos

        while self._pos < len(self._text) and self._text[self._pos] != quote:
            if self._text[self._pos] == "\\":
                self._pos += 1  # Skip escape
            self._pos += 1

        if self._pos >= len(self._text):
            raise ValueError("Unterminated string")

        value = self._text[start : self._pos]
        self._pos += 1  # Skip closing quote
        return value

    def _parse_list(self) -> list[Any]:
        """Parse list literal."""
        self._pos += 1  # Skip '['
        items = []

        while True:
            self._skip_whitespace()

            if self._pos >= len(self._text):
                raise ValueError("Unterminated list")

            if self._text[self._pos] == "]":
                self._pos += 1
                break

            items.append(self._parse_value())

            self._skip_whitespace()
            if self._pos < len(self._text) and self._text[self._pos] == ",":
                self._pos += 1

        return items

    def _parse_number_or_identifier(self) -> Any:
        """Parse number or identifier."""
        # Match number pattern (including scientific notation)
        pattern = r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?"
        match = re.match(pattern, self._text[self._pos :])

        if match:
            self._pos += match.end()
            value = match.group()
            if "." in value or "e" in value.lower():
                return float(value)
            return int(value)

        # Try identifier (True, False, None)
        ident = self._parse_identifier()
        if ident == "True":
            return True
        elif ident == "False":
            return False
        elif ident == "None":
            return None
        return ident

    def _parse_identifier(self) -> str:
        """Parse identifier."""
        start = self._pos

        if self._pos < len(self._text) and (
            self._text[self._pos].isalpha() or self._text[self._pos] == "_"
        ):
            self._pos += 1
            while self._pos < len(self._text) and (
                self._text[self._pos].isalnum() or self._text[self._pos] == "_"
            ):
                self._pos += 1

        if self._pos == start:
            raise ValueError(f"Expected identifier at position {self._pos}")

        return self._text[start : self._pos]

    def _try_parse_identifier(self) -> str | None:
        """Try to parse identifier, return None on failure."""
        start = self._pos
        try:
            return self._parse_identifier()
        except ValueError:
            self._pos = start
            return None

    def _skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self._pos < len(self._text) and self._text[self._pos].isspace():
            self._pos += 1


class DSLExecutor:
    """Executes parsed DSL expressions.

    References:
        API-010: Domain-Specific Language (DSL)
    """

    def __init__(self) -> None:
        """Initialize executor."""
        self._operations: dict[str, Callable] = {}  # type: ignore[type-arg]
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in operations."""
        # Filter operations
        self._operations["lowpass"] = self._lowpass
        self._operations["highpass"] = self._highpass
        self._operations["bandpass"] = self._bandpass

        # Analysis operations
        self._operations["fft"] = self._fft
        self._operations["psd"] = self._psd

        # Measurement operations
        self._operations["mean"] = lambda data: np.mean(data)
        self._operations["std"] = lambda data: np.std(data)
        self._operations["min"] = lambda data: np.min(data)
        self._operations["max"] = lambda data: np.max(data)
        self._operations["rms"] = lambda data: np.sqrt(np.mean(data**2))

        # Transform operations
        self._operations["normalize"] = self._normalize
        self._operations["resample"] = self._resample
        self._operations["slice"] = self._slice

    def execute(self, expr: DSLExpression, data: NDArray[np.float64]) -> Any:
        """Execute DSL expression on data.

        Args:
            expr: Parsed expression
            data: Input data

        Returns:
            Result of execution

        Raises:
            ValueError: If operation is unknown or result cannot be chained.
        """
        # Execute operation
        op = self._operations.get(expr.operation)
        if op is None:
            raise ValueError(f"Unknown operation: {expr.operation}")

        result = op(data, *expr.args, **expr.kwargs)

        # Execute chain if present
        if expr.chain:
            if isinstance(result, np.ndarray):
                return self.execute(expr.chain, result)
            else:
                raise ValueError(f"Cannot chain after {expr.operation}: result is not an array")

        return result

    def _lowpass(
        self, data: NDArray[np.float64], cutoff: float = 1e6, **kwargs: Any
    ) -> NDArray[np.float64]:
        """Low-pass filter."""
        from scipy import signal

        b, a = signal.butter(4, cutoff, btype="low", fs=kwargs.get("fs", 2 * cutoff))
        result: NDArray[np.float64] = signal.filtfilt(b, a, data)
        return result

    def _highpass(
        self, data: NDArray[np.float64], cutoff: float = 1e3, **kwargs: Any
    ) -> NDArray[np.float64]:
        """High-pass filter."""
        from scipy import signal

        b, a = signal.butter(4, cutoff, btype="high", fs=kwargs.get("fs", 10 * cutoff))
        result: NDArray[np.float64] = signal.filtfilt(b, a, data)
        return result

    def _bandpass(
        self,
        data: NDArray[np.float64],
        low: float = 1e3,
        high: float = 1e6,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        """Band-pass filter."""
        from scipy import signal

        b, a = signal.butter(4, [low, high], btype="band", fs=kwargs.get("fs", 2 * high))
        result: NDArray[np.float64] = signal.filtfilt(b, a, data)
        return result

    def _fft(
        self,
        data: NDArray[np.float64],
        nfft: int | None = None,
        **kwargs: Any,
    ) -> NDArray[np.complex128]:
        """FFT."""
        return np.fft.fft(data, n=nfft)

    def _psd(
        self,
        data: NDArray[np.float64],
        nperseg: int = 256,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        """Power spectral density."""
        from scipy import signal

        _, psd_result = signal.welch(data, nperseg=nperseg)
        result: NDArray[np.float64] = psd_result
        return result

    def _normalize(
        self,
        data: NDArray[np.float64],
        method: str = "minmax",
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        """Normalize data."""
        if method == "minmax":
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max - data_min > 0:
                result: NDArray[np.float64] = (data - data_min) / (data_max - data_min)
                return result
            return data
        elif method == "zscore":
            std = np.std(data)
            if std > 0:
                result_z: NDArray[np.float64] = (data - np.mean(data)) / std
                return result_z
            result_mean: NDArray[np.float64] = data - np.mean(data)
            return result_mean
        return data

    def _resample(
        self,
        data: NDArray[np.float64],
        factor: int = 2,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        """Resample data."""
        from scipy import signal

        result: NDArray[np.float64] = signal.resample(data, len(data) // factor)
        return result

    def _slice(
        self,
        data: NDArray[np.float64],
        start: int = 0,
        end: int | None = None,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        """Slice data."""
        return data[start:end]


# Global parser and executor
_parser = DSLParser()
_executor = DSLExecutor()


def parse_expression(text: str) -> DSLExpression:
    """Parse DSL expression.

    Args:
        text: DSL expression text

    Returns:
        Parsed expression

    Example:
        >>> expr = parse_expression("lowpass(cutoff=1e6) | fft(nfft=8192)")

    References:
        API-010: Domain-Specific Language (DSL)
    """
    return _parser.parse(text)


def analyze(data: NDArray[np.float64], expression: str) -> Any:
    """Analyze data using DSL expression.

    Args:
        data: Input data array
        expression: DSL expression string

    Returns:
        Analysis result

    Example:
        >>> result = analyze(data, "lowpass(cutoff=1e6) | fft(nfft=8192)")

    References:
        API-010: Domain-Specific Language (DSL)
    """
    expr = parse_expression(expression)
    return _executor.execute(expr, data)
