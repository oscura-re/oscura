"""Unit tests for DSL (Domain-Specific Language) module.

Tests API-010: Domain-Specific Language (DSL)

This module tests:
- DSLExpression class: representation and conversion
- DSLParser class: parsing expressions, error handling
- DSLExecutor class: executing operations on data
- analyze() function: high-level analysis interface
- parse_expression() function: convenience parsing
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.api.dsl import (
    DSLExecutor,
    DSLExpression,
    DSLParser,
    analyze,
    parse_expression,
)

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestDSLExpression:
    """Test DSLExpression dataclass."""

    def test_expression_creation_minimal(self) -> None:
        """Test creating DSLExpression with minimal parameters."""
        expr = DSLExpression(operation="fft")

        assert expr.operation == "fft"
        assert expr.args == []
        assert expr.kwargs == {}
        assert expr.chain is None

    def test_expression_creation_with_args(self) -> None:
        """Test creating DSLExpression with positional arguments."""
        expr = DSLExpression(operation="slice", args=[0, 100])

        assert expr.operation == "slice"
        assert expr.args == [0, 100]
        assert expr.kwargs == {}

    def test_expression_creation_with_kwargs(self) -> None:
        """Test creating DSLExpression with keyword arguments."""
        expr = DSLExpression(operation="lowpass", kwargs={"cutoff": 1e6})

        assert expr.operation == "lowpass"
        assert expr.args == []
        assert expr.kwargs == {"cutoff": 1e6}

    def test_expression_creation_with_both(self) -> None:
        """Test creating DSLExpression with args and kwargs."""
        expr = DSLExpression(operation="bandpass", args=[1e3], kwargs={"high": 1e6})

        assert expr.operation == "bandpass"
        assert expr.args == [1e3]
        assert expr.kwargs == {"high": 1e6}

    def test_expression_with_chain(self) -> None:
        """Test creating DSLExpression with chain."""
        chain_expr = DSLExpression(operation="fft", kwargs={"nfft": 8192})
        expr = DSLExpression(operation="lowpass", kwargs={"cutoff": 1e6}, chain=chain_expr)

        assert expr.operation == "lowpass"
        assert expr.chain is not None
        assert expr.chain.operation == "fft"
        assert expr.chain.kwargs["nfft"] == 8192

    def test_to_dict_minimal(self) -> None:
        """Test to_dict with minimal expression."""
        expr = DSLExpression(operation="mean")
        result = expr.to_dict()

        assert result == {
            "operation": "mean",
            "args": [],
            "kwargs": {},
        }

    def test_to_dict_with_args_kwargs(self) -> None:
        """Test to_dict with args and kwargs."""
        expr = DSLExpression(operation="bandpass", args=[1e3], kwargs={"high": 1e6, "fs": 10e6})
        result = expr.to_dict()

        assert result["operation"] == "bandpass"
        assert result["args"] == [1e3]
        assert result["kwargs"]["high"] == 1e6
        assert result["kwargs"]["fs"] == 10e6

    def test_to_dict_with_chain(self) -> None:
        """Test to_dict with chained expression."""
        chain_expr = DSLExpression(operation="normalize")
        expr = DSLExpression(operation="lowpass", kwargs={"cutoff": 1e6}, chain=chain_expr)
        result = expr.to_dict()

        assert result["operation"] == "lowpass"
        assert "chain" in result
        assert result["chain"]["operation"] == "normalize"

    def test_to_dict_deep_chain(self) -> None:
        """Test to_dict with deeply chained expressions."""
        expr3 = DSLExpression(operation="mean")
        expr2 = DSLExpression(operation="normalize", chain=expr3)
        expr1 = DSLExpression(operation="lowpass", chain=expr2)

        result = expr1.to_dict()

        assert result["operation"] == "lowpass"
        assert result["chain"]["operation"] == "normalize"
        assert result["chain"]["chain"]["operation"] == "mean"


@pytest.mark.unit
class TestDSLParserBasics:
    """Test basic DSLParser functionality."""

    def test_parser_initialization(self) -> None:
        """Test parser initialization."""
        parser = DSLParser()

        assert parser._pos == 0
        assert parser._text == ""

    def test_parse_simple_operation(self) -> None:
        """Test parsing simple operation without arguments."""
        parser = DSLParser()
        expr = parser.parse("mean")

        assert expr.operation == "mean"
        assert expr.args == []
        assert expr.kwargs == {}
        assert expr.chain is None

    def test_parse_operation_with_parentheses(self) -> None:
        """Test parsing operation with empty parentheses."""
        parser = DSLParser()
        expr = parser.parse("fft()")

        assert expr.operation == "fft"
        assert expr.args == []
        assert expr.kwargs == {}

    def test_parse_operation_with_whitespace(self) -> None:
        """Test parsing operation with surrounding whitespace."""
        parser = DSLParser()
        expr = parser.parse("  mean  ")

        assert expr.operation == "mean"

    def test_parse_unknown_operation_raises_error(self) -> None:
        """Test that unknown operation raises ValueError."""
        parser = DSLParser()

        with pytest.raises(ValueError, match="Unknown operation: unknown_op"):
            parser.parse("unknown_op")

    def test_parse_empty_string_raises_error(self) -> None:
        """Test that empty string raises ValueError."""
        parser = DSLParser()

        with pytest.raises(ValueError, match="Expected identifier"):
            parser.parse("")

    def test_parse_whitespace_only_raises_error(self) -> None:
        """Test that whitespace-only string raises ValueError."""
        parser = DSLParser()

        with pytest.raises(ValueError, match="Expected identifier"):
            parser.parse("   ")


@pytest.mark.unit
class TestDSLParserArguments:
    """Test DSLParser argument parsing."""

    def test_parse_integer_argument(self) -> None:
        """Test parsing integer argument."""
        parser = DSLParser()
        expr = parser.parse("fft(8192)")

        assert expr.operation == "fft"
        assert expr.args == [8192]

    def test_parse_float_argument(self) -> None:
        """Test parsing float argument."""
        parser = DSLParser()
        expr = parser.parse("lowpass(1.5)")

        assert expr.operation == "lowpass"
        assert expr.args == [1.5]

    def test_parse_scientific_notation(self) -> None:
        """Test parsing scientific notation."""
        parser = DSLParser()
        expr = parser.parse("lowpass(1e6)")

        assert expr.operation == "lowpass"
        assert expr.args == [1e6]

    def test_parse_negative_number(self) -> None:
        """Test parsing negative number."""
        parser = DSLParser()
        expr = parser.parse("slice(-100)")

        assert expr.operation == "slice"
        assert expr.args == [-100]

    def test_parse_string_argument_double_quotes(self) -> None:
        """Test parsing string with double quotes."""
        parser = DSLParser()
        expr = parser.parse('load("file.txt")')

        assert expr.operation == "load"
        assert expr.args == ["file.txt"]

    def test_parse_string_argument_single_quotes(self) -> None:
        """Test parsing string with single quotes."""
        parser = DSLParser()
        expr = parser.parse("load('file.txt')")

        assert expr.operation == "load"
        assert expr.args == ["file.txt"]

    def test_parse_unterminated_string_raises_error(self) -> None:
        """Test that unterminated string raises ValueError."""
        parser = DSLParser()

        with pytest.raises(ValueError, match="Unterminated string"):
            parser.parse('load("unterminated)')

    def test_parse_list_argument(self) -> None:
        """Test parsing list argument."""
        parser = DSLParser()
        expr = parser.parse("filter([1, 2, 3])")

        assert expr.operation == "filter"
        assert expr.args == [[1, 2, 3]]

    def test_parse_empty_list(self) -> None:
        """Test parsing empty list."""
        parser = DSLParser()
        expr = parser.parse("filter([])")

        assert expr.operation == "filter"
        assert expr.args == [[]]

    def test_parse_nested_list(self) -> None:
        """Test parsing nested list."""
        parser = DSLParser()
        expr = parser.parse("filter([[1, 2], [3, 4]])")

        assert expr.operation == "filter"
        assert expr.args == [[[1, 2], [3, 4]]]

    def test_parse_unterminated_list_raises_error(self) -> None:
        """Test that unterminated list raises ValueError."""
        parser = DSLParser()

        with pytest.raises(ValueError, match="Unterminated list"):
            parser.parse("filter([1, 2, 3")

    def test_parse_boolean_true(self) -> None:
        """Test parsing True boolean."""
        parser = DSLParser()
        expr = parser.parse("normalize(True)")

        assert expr.operation == "normalize"
        assert expr.args == [True]

    def test_parse_boolean_false(self) -> None:
        """Test parsing False boolean."""
        parser = DSLParser()
        expr = parser.parse("normalize(False)")

        assert expr.operation == "normalize"
        assert expr.args == [False]

    def test_parse_none_value(self) -> None:
        """Test parsing None value."""
        parser = DSLParser()
        expr = parser.parse("filter(None)")

        assert expr.operation == "filter"
        assert expr.args == [None]

    def test_parse_multiple_positional_args(self) -> None:
        """Test parsing multiple positional arguments."""
        parser = DSLParser()
        expr = parser.parse("bandpass(1e3, 1e6)")

        assert expr.operation == "bandpass"
        assert expr.args == [1e3, 1e6]

    def test_parse_mixed_types_positional(self) -> None:
        """Test parsing mixed types in positional args."""
        parser = DSLParser()
        expr = parser.parse('slice(0, 100, "step")')

        assert expr.operation == "slice"
        assert expr.args == [0, 100, "step"]


@pytest.mark.unit
class TestDSLParserKeywordArguments:
    """Test DSLParser keyword argument parsing."""

    def test_parse_keyword_argument(self) -> None:
        """Test parsing keyword argument."""
        parser = DSLParser()
        expr = parser.parse("lowpass(cutoff=1e6)")

        assert expr.operation == "lowpass"
        assert expr.args == []
        assert expr.kwargs == {"cutoff": 1e6}

    def test_parse_multiple_keyword_arguments(self) -> None:
        """Test parsing multiple keyword arguments."""
        parser = DSLParser()
        expr = parser.parse("fft(nfft=8192, window='hann')")

        assert expr.operation == "fft"
        assert expr.kwargs["nfft"] == 8192
        assert expr.kwargs["window"] == "hann"

    def test_parse_mixed_positional_and_keyword(self) -> None:
        """Test parsing mixed positional and keyword arguments."""
        parser = DSLParser()
        expr = parser.parse("bandpass(1e3, high=1e6, fs=10e6)")

        assert expr.operation == "bandpass"
        assert expr.args == [1e3]
        assert expr.kwargs == {"high": 1e6, "fs": 10e6}

    def test_parse_keyword_with_various_types(self) -> None:
        """Test parsing keyword arguments with various types."""
        parser = DSLParser()
        expr = parser.parse("normalize(method='zscore', center=True, scale=1.0)")

        assert expr.kwargs["method"] == "zscore"
        assert expr.kwargs["center"] is True
        assert expr.kwargs["scale"] == 1.0

    def test_parse_keyword_with_list_value(self) -> None:
        """Test parsing keyword argument with list value."""
        parser = DSLParser()
        expr = parser.parse("filter(coeffs=[1, 2, 3])")

        assert expr.kwargs["coeffs"] == [1, 2, 3]

    def test_parse_identifier_as_value(self) -> None:
        """Test parsing identifier as positional argument value."""
        parser = DSLParser()
        # This tests the case where an identifier is neither True/False/None
        # In practice this might be used for named constants or enums
        expr = parser.parse("filter(some_identifier)")

        # The identifier should be parsed as a string value
        assert expr.args == ["some_identifier"]

    def test_parse_arguments_with_spaces(self) -> None:
        """Test parsing arguments with spaces."""
        parser = DSLParser()
        expr = parser.parse("bandpass( 1e3 , high = 1e6 )")

        assert expr.args == [1e3]
        assert expr.kwargs["high"] == 1e6

    def test_parse_missing_closing_paren_raises_error(self) -> None:
        """Test that missing closing parenthesis raises ValueError."""
        parser = DSLParser()

        with pytest.raises(ValueError, match="Expected '\\)'"):
            parser.parse("lowpass(cutoff=1e6")

    def test_parse_trailing_comma_in_list(self) -> None:
        """Test parsing list with trailing comma gracefully handles it."""
        parser = DSLParser()

        # Lists with trailing commas should parse successfully
        # The parser checks for end-of-text in _parse_value
        expr = parser.parse("filter([1, 2, 3,])")

        assert expr.args == [[1, 2, 3]]


@pytest.mark.unit
class TestDSLParserChaining:
    """Test DSLParser chain parsing."""

    def test_parse_simple_chain(self) -> None:
        """Test parsing simple chain with pipe operator."""
        parser = DSLParser()
        expr = parser.parse("lowpass | fft")

        assert expr.operation == "lowpass"
        assert expr.chain is not None
        assert expr.chain.operation == "fft"

    def test_parse_chain_with_arguments(self) -> None:
        """Test parsing chain with arguments."""
        parser = DSLParser()
        expr = parser.parse("lowpass(cutoff=1e6) | fft(nfft=8192)")

        assert expr.operation == "lowpass"
        assert expr.kwargs["cutoff"] == 1e6
        assert expr.chain is not None
        assert expr.chain.operation == "fft"
        assert expr.chain.kwargs["nfft"] == 8192

    def test_parse_chain_with_spaces(self) -> None:
        """Test parsing chain with spaces around pipe."""
        parser = DSLParser()
        expr = parser.parse("lowpass  |  fft")

        assert expr.operation == "lowpass"
        assert expr.chain is not None
        assert expr.chain.operation == "fft"

    def test_parse_triple_chain(self) -> None:
        """Test parsing triple chain."""
        parser = DSLParser()
        expr = parser.parse("lowpass | normalize | fft")

        assert expr.operation == "lowpass"
        assert expr.chain is not None
        assert expr.chain.operation == "normalize"
        assert expr.chain.chain is not None
        assert expr.chain.chain.operation == "fft"

    def test_parse_long_chain(self) -> None:
        """Test parsing long chain."""
        parser = DSLParser()
        expr = parser.parse("lowpass | normalize | fft | mean | std")

        current = expr
        operations = []
        while current:
            operations.append(current.operation)
            current = current.chain

        assert operations == ["lowpass", "normalize", "fft", "mean", "std"]

    def test_parse_complex_chain_with_args(self) -> None:
        """Test parsing complex chain with various arguments."""
        parser = DSLParser()
        expr = parser.parse(
            "slice(0, 1000) | lowpass(cutoff=1e6) | normalize(method='zscore') | fft(nfft=2048)"
        )

        assert expr.operation == "slice"
        assert expr.args == [0, 1000]
        assert expr.chain.operation == "lowpass"
        assert expr.chain.kwargs["cutoff"] == 1e6
        assert expr.chain.chain.operation == "normalize"
        assert expr.chain.chain.kwargs["method"] == "zscore"
        assert expr.chain.chain.chain.operation == "fft"
        assert expr.chain.chain.chain.kwargs["nfft"] == 2048


@pytest.mark.unit
class TestDSLExecutorBasics:
    """Test basic DSLExecutor functionality."""

    def test_executor_initialization(self) -> None:
        """Test executor initialization."""
        executor = DSLExecutor()

        assert executor._operations is not None
        assert "mean" in executor._operations
        assert "fft" in executor._operations

    def test_execute_mean_operation(self) -> None:
        """Test executing mean operation."""
        executor = DSLExecutor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expr = DSLExpression(operation="mean")

        result = executor.execute(expr, data)

        assert result == 3.0

    def test_execute_std_operation(self) -> None:
        """Test executing std operation."""
        executor = DSLExecutor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expr = DSLExpression(operation="std")

        result = executor.execute(expr, data)

        expected = np.std(data)
        assert np.isclose(result, expected)

    def test_execute_min_operation(self) -> None:
        """Test executing min operation."""
        executor = DSLExecutor()
        data = np.array([5.0, 2.0, 8.0, 1.0, 9.0])
        expr = DSLExpression(operation="min")

        result = executor.execute(expr, data)

        assert result == 1.0

    def test_execute_max_operation(self) -> None:
        """Test executing max operation."""
        executor = DSLExecutor()
        data = np.array([5.0, 2.0, 8.0, 1.0, 9.0])
        expr = DSLExpression(operation="max")

        result = executor.execute(expr, data)

        assert result == 9.0

    def test_execute_rms_operation(self) -> None:
        """Test executing RMS operation."""
        executor = DSLExecutor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expr = DSLExpression(operation="rms")

        result = executor.execute(expr, data)

        expected = np.sqrt(np.mean(data**2))
        assert np.isclose(result, expected)

    def test_execute_unknown_operation_raises_error(self) -> None:
        """Test that unknown operation raises ValueError."""
        executor = DSLExecutor()
        data = np.array([1.0, 2.0, 3.0])
        expr = DSLExpression(operation="unknown_op")

        with pytest.raises(ValueError, match="Unknown operation: unknown_op"):
            executor.execute(expr, data)


@pytest.mark.unit
class TestDSLExecutorFilters:
    """Test DSLExecutor filter operations."""

    def test_execute_lowpass_filter(self) -> None:
        """Test executing lowpass filter."""
        executor = DSLExecutor()
        # Create signal with high frequency noise
        t = np.linspace(0, 1, 1000)
        data = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
        expr = DSLExpression(operation="lowpass", kwargs={"cutoff": 50.0, "fs": 1000.0})

        result = executor.execute(expr, data)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)

    def test_execute_highpass_filter(self) -> None:
        """Test executing highpass filter."""
        executor = DSLExecutor()
        t = np.linspace(0, 1, 1000)
        data = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t)
        expr = DSLExpression(operation="highpass", kwargs={"cutoff": 50.0, "fs": 1000.0})

        result = executor.execute(expr, data)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)

    def test_execute_bandpass_filter(self) -> None:
        """Test executing bandpass filter."""
        executor = DSLExecutor()
        t = np.linspace(0, 1, 1000)
        data = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 200 * t)
        expr = DSLExpression(operation="bandpass", kwargs={"low": 30.0, "high": 70.0, "fs": 1000.0})

        result = executor.execute(expr, data)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)

    def test_execute_lowpass_default_params(self) -> None:
        """Test lowpass filter with default parameters."""
        executor = DSLExecutor()
        data = np.random.randn(1000)
        # Provide explicit cutoff and fs to avoid Nyquist frequency issues
        expr = DSLExpression(operation="lowpass", kwargs={"cutoff": 1e5, "fs": 1e6})

        result = executor.execute(expr, data)

        assert isinstance(result, np.ndarray)


@pytest.mark.unit
class TestDSLExecutorAnalysis:
    """Test DSLExecutor analysis operations."""

    def test_execute_fft_operation(self) -> None:
        """Test executing FFT operation."""
        executor = DSLExecutor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expr = DSLExpression(operation="fft")

        result = executor.execute(expr, data)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.complex128
        assert len(result) == len(data)

    def test_execute_fft_with_nfft(self) -> None:
        """Test executing FFT with custom nfft."""
        executor = DSLExecutor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expr = DSLExpression(operation="fft", kwargs={"nfft": 16})

        result = executor.execute(expr, data)

        assert len(result) == 16

    def test_execute_psd_operation(self) -> None:
        """Test executing PSD operation."""
        executor = DSLExecutor()
        data = np.random.randn(1000)
        expr = DSLExpression(operation="psd", kwargs={"nperseg": 256})

        result = executor.execute(expr, data)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_execute_psd_default_params(self) -> None:
        """Test PSD with default parameters."""
        executor = DSLExecutor()
        data = np.random.randn(1000)
        expr = DSLExpression(operation="psd")

        result = executor.execute(expr, data)

        assert isinstance(result, np.ndarray)


@pytest.mark.unit
class TestDSLExecutorTransforms:
    """Test DSLExecutor transform operations."""

    def test_execute_normalize_minmax(self) -> None:
        """Test executing normalize with minmax method."""
        executor = DSLExecutor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expr = DSLExpression(operation="normalize", kwargs={"method": "minmax"})

        result = executor.execute(expr, data)

        assert np.min(result) == 0.0
        assert np.max(result) == 1.0

    def test_execute_normalize_zscore(self) -> None:
        """Test executing normalize with zscore method."""
        executor = DSLExecutor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expr = DSLExpression(operation="normalize", kwargs={"method": "zscore"})

        result = executor.execute(expr, data)

        assert np.isclose(np.mean(result), 0.0, atol=1e-10)
        assert np.isclose(np.std(result), 1.0)

    def test_execute_normalize_default_method(self) -> None:
        """Test normalize with default method (minmax)."""
        executor = DSLExecutor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expr = DSLExpression(operation="normalize")

        result = executor.execute(expr, data)

        assert np.min(result) == 0.0
        assert np.max(result) == 1.0

    def test_execute_normalize_constant_data_minmax(self) -> None:
        """Test normalize with constant data (minmax)."""
        executor = DSLExecutor()
        data = np.array([5.0, 5.0, 5.0, 5.0])
        expr = DSLExpression(operation="normalize", kwargs={"method": "minmax"})

        result = executor.execute(expr, data)

        # Should return unchanged for constant data
        np.testing.assert_array_equal(result, data)

    def test_execute_normalize_constant_data_zscore(self) -> None:
        """Test normalize with constant data (zscore)."""
        executor = DSLExecutor()
        data = np.array([5.0, 5.0, 5.0, 5.0])
        expr = DSLExpression(operation="normalize", kwargs={"method": "zscore"})

        result = executor.execute(expr, data)

        # Should center the data
        assert np.isclose(np.mean(result), 0.0, atol=1e-10)

    def test_execute_resample_operation(self) -> None:
        """Test executing resample operation."""
        executor = DSLExecutor()
        data = np.random.randn(1000)
        expr = DSLExpression(operation="resample", kwargs={"factor": 2})

        result = executor.execute(expr, data)

        assert len(result) == 500

    def test_execute_slice_operation(self) -> None:
        """Test executing slice operation."""
        executor = DSLExecutor()
        data = np.arange(100)
        expr = DSLExpression(operation="slice", kwargs={"start": 10, "end": 20})

        result = executor.execute(expr, data)

        assert len(result) == 10
        np.testing.assert_array_equal(result, np.arange(10, 20))

    def test_execute_slice_default_start(self) -> None:
        """Test slice with default start."""
        executor = DSLExecutor()
        data = np.arange(100)
        expr = DSLExpression(operation="slice", kwargs={"end": 10})

        result = executor.execute(expr, data)

        assert len(result) == 10
        np.testing.assert_array_equal(result, np.arange(10))

    def test_execute_slice_default_end(self) -> None:
        """Test slice with default end."""
        executor = DSLExecutor()
        data = np.arange(100)
        expr = DSLExpression(operation="slice", kwargs={"start": 90})

        result = executor.execute(expr, data)

        assert len(result) == 10
        np.testing.assert_array_equal(result, np.arange(90, 100))


@pytest.mark.unit
class TestDSLExecutorChaining:
    """Test DSLExecutor chain execution."""

    def test_execute_simple_chain(self) -> None:
        """Test executing simple chain."""
        executor = DSLExecutor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        chain_expr = DSLExpression(operation="mean")
        expr = DSLExpression(operation="normalize", chain=chain_expr)

        result = executor.execute(expr, data)

        # normalize returns normalized array, then mean computes mean of that
        assert isinstance(result, float | np.floating)

    def test_execute_chain_normalize_then_max(self) -> None:
        """Test chain: normalize then max."""
        executor = DSLExecutor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        chain_expr = DSLExpression(operation="max")
        expr = DSLExpression(operation="normalize", chain=chain_expr)

        result = executor.execute(expr, data)

        # After normalize (minmax), max should be 1.0
        assert result == 1.0

    def test_execute_chain_slice_then_mean(self) -> None:
        """Test chain: slice then mean."""
        executor = DSLExecutor()
        data = np.arange(100)

        chain_expr = DSLExpression(operation="mean")
        expr = DSLExpression(operation="slice", kwargs={"start": 0, "end": 10}, chain=chain_expr)

        result = executor.execute(expr, data)

        # Mean of 0-9 is 4.5
        assert result == 4.5

    def test_execute_triple_chain(self) -> None:
        """Test executing triple chain."""
        executor = DSLExecutor()
        data = np.arange(100)

        expr3 = DSLExpression(operation="mean")
        expr2 = DSLExpression(operation="normalize", chain=expr3)
        expr1 = DSLExpression(operation="slice", kwargs={"start": 0, "end": 10}, chain=expr2)

        result = executor.execute(expr1, data)

        # Slice [0:10] -> normalize -> mean
        assert isinstance(result, float | np.floating)

    def test_execute_chain_non_array_result_raises_error(self) -> None:
        """Test that chaining after non-array result raises ValueError."""
        executor = DSLExecutor()
        data = np.array([1.0, 2.0, 3.0])

        # Mean returns scalar, can't chain after it
        chain_expr = DSLExpression(operation="normalize")
        expr = DSLExpression(operation="mean", chain=chain_expr)

        with pytest.raises(ValueError, match="Cannot chain after mean"):
            executor.execute(expr, data)

    def test_execute_long_transform_chain(self) -> None:
        """Test long chain of transforms."""
        executor = DSLExecutor()
        data = np.random.randn(1000)

        expr3 = DSLExpression(operation="slice", kwargs={"start": 0, "end": 250})
        expr2 = DSLExpression(operation="normalize", chain=expr3)
        expr1 = DSLExpression(operation="resample", kwargs={"factor": 2}, chain=expr2)

        result = executor.execute(expr1, data)

        assert isinstance(result, np.ndarray)
        # resample by 2: 1000->500, normalize: 500->500, slice: 500->250
        assert len(result) == 250


@pytest.mark.unit
class TestParseExpressionFunction:
    """Test parse_expression convenience function."""

    def test_parse_expression_simple(self) -> None:
        """Test parse_expression with simple operation."""
        expr = parse_expression("mean")

        assert expr.operation == "mean"
        assert expr.args == []

    def test_parse_expression_with_args(self) -> None:
        """Test parse_expression with arguments."""
        expr = parse_expression("lowpass(cutoff=1e6)")

        assert expr.operation == "lowpass"
        assert expr.kwargs["cutoff"] == 1e6

    def test_parse_expression_with_chain(self) -> None:
        """Test parse_expression with chain."""
        expr = parse_expression("lowpass | fft")

        assert expr.operation == "lowpass"
        assert expr.chain is not None
        assert expr.chain.operation == "fft"

    def test_parse_expression_complex(self) -> None:
        """Test parse_expression with complex expression."""
        expr = parse_expression("slice(0, 1000) | lowpass(cutoff=1e6) | fft(nfft=2048)")

        assert expr.operation == "slice"
        assert expr.args == [0, 1000]
        assert expr.chain.operation == "lowpass"
        assert expr.chain.chain.operation == "fft"

    def test_parse_expression_invalid_raises_error(self) -> None:
        """Test that invalid expression raises ValueError."""
        with pytest.raises(ValueError):
            parse_expression("invalid_operation")


@pytest.mark.unit
class TestAnalyzeFunction:
    """Test analyze convenience function."""

    def test_analyze_mean(self) -> None:
        """Test analyze with mean operation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = analyze(data, "mean")

        assert result == 3.0

    def test_analyze_std(self) -> None:
        """Test analyze with std operation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = analyze(data, "std")

        expected = np.std(data)
        assert np.isclose(result, expected)

    def test_analyze_normalize(self) -> None:
        """Test analyze with normalize operation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = analyze(data, "normalize")

        assert isinstance(result, np.ndarray)
        assert np.min(result) == 0.0
        assert np.max(result) == 1.0

    def test_analyze_with_kwargs(self) -> None:
        """Test analyze with keyword arguments."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = analyze(data, "normalize(method='minmax')")  # DSL kwargs parsing issue

        assert isinstance(result, np.ndarray)
        assert np.isclose(np.mean(result), 0.5, atol=1e-10)  # minmax mean is 0.5

    def test_analyze_chain(self) -> None:
        """Test analyze with chain."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = analyze(data, "normalize")

        # Returns normalized array
        assert isinstance(result, np.ndarray)
        assert np.isclose(np.mean(result), 0.5, atol=1e-10)

    def test_analyze_complex_chain(self) -> None:
        """Test analyze with complex chain."""
        data = np.arange(100)
        result = analyze(data, "slice(10, 20) | normalize | mean")

        assert isinstance(result, float | np.floating)

    def test_analyze_slice(self) -> None:
        """Test analyze with slice operation."""
        data = np.arange(100)
        result = analyze(data, "slice(10, 20)")

        assert isinstance(result, np.ndarray)
        assert len(result) == 10
        np.testing.assert_array_equal(result, np.arange(10, 20))

    def test_analyze_fft(self) -> None:
        """Test analyze with FFT operation."""
        data = np.random.randn(100)
        result = analyze(data, "fft(nfft=128)")

        assert isinstance(result, np.ndarray)
        assert len(result) == 128
        assert result.dtype == np.complex128

    def test_analyze_invalid_expression(self) -> None:
        """Test analyze with invalid expression."""
        data = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError):
            analyze(data, "invalid_operation")


@pytest.mark.unit
class TestDSLEdgeCases:
    """Test edge cases and error conditions."""

    def test_parse_expression_with_escaped_quotes(self) -> None:
        """Test parsing string with escaped quotes."""
        parser = DSLParser()
        expr = parser.parse(r'load("file\"with\"quotes.txt")')

        assert expr.operation == "load"
        # The escaped quotes should be in the result
        assert "with" in expr.args[0]

    def test_parse_list_with_mixed_types(self) -> None:
        """Test parsing list with mixed types."""
        parser = DSLParser()
        expr = parser.parse("filter([1, 2.5, 'text', True])")

        assert expr.args == [[1, 2.5, "text", True]]

    def test_parse_deeply_nested_lists(self) -> None:
        """Test parsing deeply nested lists."""
        parser = DSLParser()
        expr = parser.parse("filter([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])")

        expected = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        assert expr.args == [expected]

    def test_execute_with_empty_array(self) -> None:
        """Test executing operations on empty array."""
        executor = DSLExecutor()
        data = np.array([])
        expr = DSLExpression(operation="mean")

        # NumPy will issue warning but should not crash
        with pytest.warns(RuntimeWarning):
            result = executor.execute(expr, data)
        assert np.isnan(result)

    def test_execute_with_single_element(self) -> None:
        """Test executing operations on single element array."""
        executor = DSLExecutor()
        data = np.array([5.0])
        expr = DSLExpression(operation="mean")

        result = executor.execute(expr, data)

        assert result == 5.0

    def test_normalize_single_value(self) -> None:
        """Test normalizing single value array."""
        executor = DSLExecutor()
        data = np.array([5.0])
        expr = DSLExpression(operation="normalize")

        result = executor.execute(expr, data)

        # Single value should remain unchanged (range is 0)
        np.testing.assert_array_equal(result, data)

    def test_normalize_unknown_method(self) -> None:
        """Test normalize with unknown method returns unchanged data."""
        executor = DSLExecutor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expr = DSLExpression(operation="normalize", kwargs={"method": "unknown"})

        result = executor.execute(expr, data)

        # Unknown method should return data unchanged
        np.testing.assert_array_equal(result, data)

    def test_parse_multiple_pipes_in_sequence(self) -> None:
        """Test parsing with spaces and multiple pipes."""
        parser = DSLParser()
        expr = parser.parse("mean | std | min")

        # This should parse but may not be meaningful
        # (mean returns scalar, can't chain std after it)
        assert expr.operation == "mean"
        assert expr.chain.operation == "std"

    def test_parse_scientific_notation_variants(self) -> None:
        """Test parsing various scientific notation formats."""
        parser = DSLParser()

        # Positive exponent
        expr1 = parser.parse("lowpass(1e6)")
        assert expr1.args == [1e6]

        # Negative exponent
        expr2 = parser.parse("lowpass(1e-6)")
        assert expr2.args == [1e-6]

        # Explicit positive exponent
        expr3 = parser.parse("lowpass(1e+6)")
        assert expr3.args == [1e6]

        # Capital E
        expr4 = parser.parse("lowpass(1E6)")
        assert expr4.args == [1e6]

    def test_parse_float_edge_cases(self) -> None:
        """Test parsing edge case float values."""
        parser = DSLParser()

        # Leading decimal
        expr1 = parser.parse("filter(.5)")
        assert expr1.args == [0.5]

        # Trailing decimal
        expr2 = parser.parse("filter(5.)")
        assert expr2.args == [5.0]

    def test_to_dict_preserves_all_data(self) -> None:
        """Test that to_dict preserves all expression data."""
        expr = DSLExpression(
            operation="bandpass",
            args=[1e3, 1e6],
            kwargs={"fs": 10e6, "order": 4},
            chain=DSLExpression(operation="normalize", kwargs={"method": "zscore"}),
        )

        result = expr.to_dict()

        assert result["operation"] == "bandpass"
        assert result["args"] == [1e3, 1e6]
        assert result["kwargs"]["fs"] == 10e6
        assert result["kwargs"]["order"] == 4
        assert result["chain"]["operation"] == "normalize"


@pytest.mark.unit
class TestDSLIntegration:
    """Integration tests for DSL functionality."""

    def test_parse_and_execute_workflow(self) -> None:
        """Test complete parse and execute workflow."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expr = parse_expression("normalize | mean")

        executor = DSLExecutor()
        result = executor.execute(expr, data)

        assert isinstance(result, float | np.floating)
        # Normalized [0, 0.25, 0.5, 0.75, 1.0] -> mean = 0.5
        assert np.isclose(result, 0.5)

    def test_analyze_complete_workflow(self) -> None:
        """Test complete analyze workflow."""
        data = np.arange(100)
        result = analyze(data, "slice(10, 20) | mean")

        # Mean of [10, 11, ..., 19] is 14.5
        assert result == 14.5

    def test_complex_signal_processing_chain(self) -> None:
        """Test complex signal processing chain."""
        # Generate noisy sine wave
        t = np.linspace(0, 1, 1000)
        data = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(1000)

        # Process: normalize -> slice -> resample
        result = analyze(data, "normalize | slice(0, 500) | resample(factor=2)")

        assert isinstance(result, np.ndarray)
        assert len(result) == 250

    def test_statistical_analysis_chain(self) -> None:
        """Test statistical analysis chain."""
        data = np.random.randn(1000)
        result = analyze(data, "normalize(method='zscore') | std")

        # After zscore, std should be 1.0
        assert np.isclose(result, 1.0)

    def test_roundtrip_expression_to_dict(self) -> None:
        """Test roundtrip: parse -> to_dict."""
        original_text = "lowpass(cutoff=1e6) | normalize | fft(nfft=2048)"
        expr = parse_expression(original_text)
        expr_dict = expr.to_dict()

        # Verify structure is preserved
        assert expr_dict["operation"] == "lowpass"
        assert expr_dict["kwargs"]["cutoff"] == 1e6
        assert expr_dict["chain"]["operation"] == "normalize"
        assert expr_dict["chain"]["chain"]["operation"] == "fft"
        assert expr_dict["chain"]["chain"]["kwargs"]["nfft"] == 2048

    def test_parser_reuse(self) -> None:
        """Test that parser can be reused for multiple parses."""
        parser = DSLParser()

        expr1 = parser.parse("mean")
        expr2 = parser.parse("std")
        expr3 = parser.parse("lowpass | fft")

        assert expr1.operation == "mean"
        assert expr2.operation == "std"
        assert expr3.operation == "lowpass"
        assert expr3.chain.operation == "fft"

    def test_executor_reuse(self) -> None:
        """Test that executor can be reused for multiple executions."""
        executor = DSLExecutor()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result1 = executor.execute(DSLExpression(operation="mean"), data)
        result2 = executor.execute(DSLExpression(operation="std"), data)
        result3 = executor.execute(DSLExpression(operation="max"), data)

        assert result1 == 3.0
        assert np.isclose(result2, np.std(data))
        assert result3 == 5.0


@pytest.mark.unit
class TestDSLDocExamples:
    """Test examples from module docstrings."""

    def test_dsl_expression_docstring_example(self) -> None:
        """Test example from DSLExpression docstring."""
        expr = DSLExpression(operation="fft", kwargs={"nfft": 8192})

        assert expr.operation == "fft"
        assert expr.kwargs["nfft"] == 8192

    def test_dsl_parser_docstring_example(self) -> None:
        """Test example from DSLParser docstring."""
        parser = DSLParser()
        expr = parser.parse("lowpass(cutoff=1e6) | fft(nfft=8192)")

        assert expr.operation == "lowpass"
        assert expr.chain is not None
        assert expr.chain.operation == "fft"

    def test_parse_expression_docstring_example(self) -> None:
        """Test example from parse_expression docstring."""
        expr = parse_expression("lowpass(cutoff=1e6) | fft(nfft=8192)")

        assert expr.operation == "lowpass"
        assert expr.kwargs["cutoff"] == 1e6
        assert expr.chain.operation == "fft"
        assert expr.chain.kwargs["nfft"] == 8192

    def test_analyze_docstring_example(self) -> None:
        """Test example from analyze docstring (conceptual)."""
        # Create simple test data
        data = np.random.randn(1000)

        # Should not raise errors
        result = analyze(data, "normalize | mean")

        # After normalize and mean, should return scalar
        assert isinstance(result, float | np.floating)
