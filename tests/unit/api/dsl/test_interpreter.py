"""Unit tests for DSL interpreter module.

Tests for interpreter functionality including:
- Interpreter class: command registration, execution
- Expression evaluation: literals, variables, function calls
- Statement execution: assignments, pipelines, for loops
- Error handling: undefined variables, unknown commands
- execute_dsl convenience function
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from oscura.api.dsl.interpreter import Interpreter, InterpreterError, execute_dsl
from oscura.api.dsl.parser import (
    Assignment,
    Command,
    ForLoop,
    FunctionCall,
    Literal,
    Pipeline,
    Variable,
)

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestInterpreterInit:
    """Test Interpreter initialization."""

    def test_init_empty_environment(self) -> None:
        """Test interpreter starts with empty environment."""
        interp = Interpreter()
        assert interp.variables == {}
        assert isinstance(interp.commands, dict)
        assert len(interp.commands) > 0

    def test_builtin_commands_registered(self) -> None:
        """Test that built-in commands are registered."""
        interp = Interpreter()
        expected_commands = {"load", "filter", "measure", "plot", "export", "glob"}
        assert expected_commands.issubset(set(interp.commands.keys()))


@pytest.mark.unit
class TestCmdLoad:
    """Test _cmd_load interpreter command."""

    def test_load_missing_argument(self) -> None:
        """Test load command without arguments."""
        interp = Interpreter()
        with pytest.raises(InterpreterError, match="requires exactly 1 argument"):
            interp._cmd_load()

    def test_load_too_many_arguments(self) -> None:
        """Test load command with too many arguments."""
        interp = Interpreter()
        with pytest.raises(InterpreterError, match="requires exactly 1 argument"):
            interp._cmd_load("file1.csv", "file2.csv")

    def test_load_non_string_filename(self) -> None:
        """Test load command with non-string filename."""
        interp = Interpreter()
        with pytest.raises(InterpreterError, match="must be a string"):
            interp._cmd_load(123)

    def test_load_nonexistent_file(self) -> None:
        """Test load command with nonexistent file."""
        interp = Interpreter()
        with pytest.raises(InterpreterError, match="File not found"):
            interp._cmd_load("/nonexistent/file.csv")

    def test_load_file_success(self, tmp_path: Path) -> None:
        """Test successful file loading."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("time,voltage\n0,1\n1,2\n")

        interp = Interpreter()
        with patch("oscura.loaders.load") as mock_load:
            mock_load.return_value = {"data": np.array([1, 2, 3])}
            result = interp._cmd_load(str(test_file))
            mock_load.assert_called_once_with(str(test_file))
            assert "data" in result


@pytest.mark.unit
class TestCmdFilter:
    """Test _cmd_filter interpreter command."""

    def test_filter_missing_filter_type(self) -> None:
        """Test filter command without filter type."""
        interp = Interpreter()
        trace = np.array([1.0, 2.0, 3.0])
        with pytest.raises(InterpreterError, match="requires filter type"):
            interp._cmd_filter(trace)

    def test_filter_non_string_type(self) -> None:
        """Test filter command with non-string filter type."""
        interp = Interpreter()
        trace = np.array([1.0, 2.0, 3.0])
        with pytest.raises(InterpreterError, match="must be a string"):
            interp._cmd_filter(trace, 123)

    def test_filter_unknown_type(self) -> None:
        """Test filter command with unknown filter type."""
        interp = Interpreter()
        trace = np.array([1.0, 2.0, 3.0])
        with pytest.raises(InterpreterError, match="Unknown filter type"):
            interp._cmd_filter(trace, "invalidfilter", 1000.0)

    def test_filter_lowpass_missing_cutoff(self) -> None:
        """Test lowpass filter without cutoff frequency."""
        interp = Interpreter()
        trace = np.array([1.0, 2.0, 3.0])
        with pytest.raises(InterpreterError, match="requires cutoff frequency"):
            interp._cmd_filter(trace, "lowpass")

    def test_filter_lowpass_non_numeric_cutoff(self) -> None:
        """Test lowpass filter with non-numeric cutoff."""
        interp = Interpreter()
        trace = np.array([1.0, 2.0, 3.0])
        with pytest.raises(InterpreterError, match="must be a number"):
            interp._cmd_filter(trace, "lowpass", "invalid")

    def test_filter_lowpass_success(self) -> None:
        """Test successful lowpass filter application."""
        interp = Interpreter()
        trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch("oscura.utils.filtering.low_pass") as mock_filter:
            mock_filter.return_value = trace * 0.5
            result = interp._cmd_filter(trace, "lowpass", 1000.0)
            mock_filter.assert_called_once_with(trace, 1000.0)
            np.testing.assert_array_equal(result, trace * 0.5)

    def test_filter_highpass_success(self) -> None:
        """Test successful highpass filter application."""
        interp = Interpreter()
        trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch("oscura.utils.filtering.high_pass") as mock_filter:
            mock_filter.return_value = trace * 0.8
            result = interp._cmd_filter(trace, "highpass", 500.0)
            mock_filter.assert_called_once_with(trace, 500.0)
            np.testing.assert_array_equal(result, trace * 0.8)

    def test_filter_bandpass_success(self) -> None:
        """Test successful bandpass filter application."""
        interp = Interpreter()
        trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch("oscura.utils.filtering.band_pass") as mock_filter:
            mock_filter.return_value = trace * 0.6
            result = interp._cmd_filter(trace, "bandpass", 100.0)
            mock_filter.assert_called_once_with(trace, 100.0)
            np.testing.assert_array_equal(result, trace * 0.6)

    def test_filter_bandstop_success(self) -> None:
        """Test successful bandstop filter application."""
        interp = Interpreter()
        trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch("oscura.utils.filtering.band_stop") as mock_filter:
            mock_filter.return_value = trace * 0.9
            result = interp._cmd_filter(trace, "bandstop", 200.0)
            mock_filter.assert_called_once_with(trace, 200.0)
            np.testing.assert_array_equal(result, trace * 0.9)


@pytest.mark.unit
class TestCmdMeasure:
    """Test _cmd_measure interpreter command."""

    def test_measure_missing_measurement_name(self) -> None:
        """Test measure command without measurement name."""
        interp = Interpreter()
        trace = np.array([1.0, 2.0, 3.0])
        with pytest.raises(InterpreterError, match="requires measurement name"):
            interp._cmd_measure(trace)

    def test_measure_non_string_name(self) -> None:
        """Test measure command with non-string measurement name."""
        interp = Interpreter()
        trace = np.array([1.0, 2.0, 3.0])
        with pytest.raises(InterpreterError, match="must be a string"):
            interp._cmd_measure(trace, 123)

    def test_measure_unknown_measurement(self) -> None:
        """Test measure command with unknown measurement."""
        interp = Interpreter()
        trace = np.array([1.0, 2.0, 3.0])
        with pytest.raises(InterpreterError, match="Unknown measurement"):
            interp._cmd_measure(trace, "invalid_measurement")

    def test_measure_frequency_success(self) -> None:
        """Test successful frequency measurement."""
        interp = Interpreter()
        trace = np.sin(np.linspace(0, 4 * np.pi, 100))

        with patch("oscura.frequency") as mock_frequency:
            mock_frequency.return_value = 1000.0
            result = interp._cmd_measure(trace, "frequency")
            mock_frequency.assert_called_once_with(trace)
            assert result == 1000.0

    def test_measure_period_success(self) -> None:
        """Test successful period measurement."""
        interp = Interpreter()
        trace = np.sin(np.linspace(0, 4 * np.pi, 100))

        with patch("oscura.period") as mock_period:
            mock_period.return_value = 1.0e-3
            result = interp._cmd_measure(trace, "period")
            assert result == 1.0e-3

    def test_measure_rise_time_success(self) -> None:
        """Test successful rise_time measurement."""
        interp = Interpreter()
        trace = np.array([0.0, 0.5, 1.0])

        with patch("oscura.rise_time") as mock_rise_time:
            mock_rise_time.return_value = 0.5e-6
            result = interp._cmd_measure(trace, "rise_time")
            assert result == 0.5e-6


@pytest.mark.unit
class TestCmdPlot:
    """Test _cmd_plot interpreter command."""

    def test_plot_default(self) -> None:
        """Test plot command with default parameters."""
        interp = Interpreter()
        trace = np.array([1.0, 2.0, 3.0])

        with patch("oscura.visualization") as mock_viz:
            mock_viz.plot = Mock()
            result = interp._cmd_plot(trace)
            mock_viz.plot.assert_called_once_with(trace)
            np.testing.assert_array_equal(result, trace)

    def test_plot_with_type(self) -> None:
        """Test plot command with plot type."""
        interp = Interpreter()
        trace = np.array([1.0, 2.0, 3.0])

        with patch("oscura.visualization") as mock_viz:
            mock_viz.plot = Mock()
            result = interp._cmd_plot(trace, "waveform")
            mock_viz.plot.assert_called_once_with(trace)
            np.testing.assert_array_equal(result, trace)

    def test_plot_non_string_type(self) -> None:
        """Test plot command with non-string type."""
        interp = Interpreter()
        trace = np.array([1.0, 2.0, 3.0])
        with pytest.raises(InterpreterError, match="must be a string"):
            interp._cmd_plot(trace, 123)

    def test_plot_unknown_type(self) -> None:
        """Test plot command with unknown plot type."""
        interp = Interpreter()
        trace = np.array([1.0, 2.0, 3.0])
        with pytest.raises(InterpreterError, match="Unknown plot type"):
            interp._cmd_plot(trace, "invalid_plot_type")


@pytest.mark.unit
class TestCmdExport:
    """Test _cmd_export interpreter command."""

    def test_export_missing_format(self) -> None:
        """Test export command without format."""
        interp = Interpreter()
        data = np.array([1, 2, 3])
        with pytest.raises(InterpreterError, match="requires format"):
            interp._cmd_export(data)

    def test_export_non_string_format(self) -> None:
        """Test export command with non-string format."""
        interp = Interpreter()
        data = np.array([1, 2, 3])
        with pytest.raises(InterpreterError, match="format must be a string"):
            interp._cmd_export(data, 123)

    def test_export_non_string_filename(self) -> None:
        """Test export command with non-string filename."""
        interp = Interpreter()
        data = np.array([1, 2, 3])
        with pytest.raises(InterpreterError, match="filename must be a string"):
            interp._cmd_export(data, "json", 456)

    def test_export_raises_redesigned_error(self) -> None:
        """Test that export raises redesigned error."""
        interp = Interpreter()
        data = np.array([1, 2, 3])
        with pytest.raises(InterpreterError, match="redesigned"):
            interp._cmd_export(data, "json", "output.json")


@pytest.mark.unit
class TestCmdGlob:
    """Test _cmd_glob interpreter command."""

    def test_glob_non_string_pattern(self) -> None:
        """Test glob command with non-string pattern."""
        interp = Interpreter()
        with pytest.raises(InterpreterError, match="pattern must be a string"):
            interp._cmd_glob(123)

    def test_glob_success(self, tmp_path: Path) -> None:
        """Test successful glob pattern matching."""
        (tmp_path / "test1.csv").touch()
        (tmp_path / "test2.csv").touch()

        interp = Interpreter()
        pattern = str(tmp_path / "*.csv")
        result = interp._cmd_glob(pattern)
        assert len(result) == 2


@pytest.mark.unit
class TestEvalExpression:
    """Test eval_expression method."""

    def test_eval_literal(self) -> None:
        """Test evaluating literal values."""
        interp = Interpreter()

        # String literal
        expr = Literal(line=1, column=0, value="hello")
        assert interp.eval_expression(expr) == "hello"

        # Number literal
        expr = Literal(line=1, column=0, value=42)
        assert interp.eval_expression(expr) == 42

        # List literal
        expr = Literal(line=1, column=0, value=[1, 2, 3])
        assert interp.eval_expression(expr) == [1, 2, 3]

    def test_eval_variable_defined(self) -> None:
        """Test evaluating defined variable."""
        interp = Interpreter()
        interp.variables["test_var"] = 123

        expr = Variable(line=1, column=0, name="test_var")
        assert interp.eval_expression(expr) == 123

    def test_eval_variable_undefined(self) -> None:
        """Test evaluating undefined variable."""
        interp = Interpreter()

        expr = Variable(line=1, column=0, name="undefined_var")
        with pytest.raises(InterpreterError, match="Undefined variable"):
            interp.eval_expression(expr)

    def test_eval_function_call(self) -> None:
        """Test evaluating function call."""
        interp = Interpreter()

        with patch.object(interp, "eval_function_call") as mock_eval:
            mock_eval.return_value = 42
            func = FunctionCall(
                line=1, column=0, name="glob", args=[Literal(line=1, column=0, value="*.csv")]
            )
            result = interp.eval_expression(func)
            assert result == 42

    def test_eval_command(self) -> None:
        """Test evaluating command."""
        interp = Interpreter()

        with patch.object(interp, "eval_command") as mock_eval:
            mock_eval.return_value = "result"
            cmd = Command(
                line=1, column=0, name="load", args=[Literal(line=1, column=0, value="file.csv")]
            )
            result = interp.eval_expression(cmd)
            assert result == "result"

    def test_eval_pipeline(self) -> None:
        """Test evaluating pipeline."""
        interp = Interpreter()

        with patch.object(interp, "eval_pipeline") as mock_eval:
            mock_eval.return_value = "pipeline_result"
            pipeline = Pipeline(
                line=1, column=0, stages=[Command(line=1, column=0, name="load", args=[])]
            )
            result = interp.eval_expression(pipeline)
            assert result == "pipeline_result"


@pytest.mark.unit
class TestEvalFunctionCall:
    """Test eval_function_call method."""

    def test_function_call_unknown_function(self) -> None:
        """Test calling unknown function."""
        interp = Interpreter()
        func = FunctionCall(line=1, column=0, name="unknown_func", args=[])

        with pytest.raises(InterpreterError, match="Unknown function"):
            interp.eval_function_call(func)

    def test_function_call_glob(self, tmp_path: Path) -> None:
        """Test calling glob function."""
        (tmp_path / "test.csv").touch()

        interp = Interpreter()
        pattern = str(tmp_path / "*.csv")
        func = FunctionCall(
            line=1, column=0, name="glob", args=[Literal(line=1, column=0, value=pattern)]
        )

        result = interp.eval_function_call(func)
        assert len(result) == 1


@pytest.mark.unit
class TestEvalCommand:
    """Test eval_command method."""

    def test_command_unknown(self) -> None:
        """Test executing unknown command."""
        interp = Interpreter()
        cmd = Command(line=1, column=0, name="unknown_cmd", args=[])

        with pytest.raises(InterpreterError, match="Unknown command"):
            interp.eval_command(cmd, None)

    def test_command_without_input(self, tmp_path: Path) -> None:
        """Test executing command without piped input."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("data")

        interp = Interpreter()

        mock_load = Mock(return_value={"data": [1, 2, 3]})
        interp.commands["load"] = mock_load
        cmd = Command(
            line=1,
            column=0,
            name="load",
            args=[Literal(line=1, column=0, value=str(test_file))],
        )
        result = interp.eval_command(cmd, None)
        mock_load.assert_called_once_with(str(test_file))
        assert result == {"data": [1, 2, 3]}

    def test_command_with_input(self) -> None:
        """Test executing command with piped input."""
        interp = Interpreter()
        trace = np.array([1.0, 2.0, 3.0])

        mock_filter = Mock(return_value=trace * 0.5)
        interp.commands["filter"] = mock_filter
        cmd = Command(
            line=1,
            column=0,
            name="filter",
            args=[
                Literal(line=1, column=0, value="lowpass"),
                Literal(line=1, column=0, value=1000.0),
            ],
        )
        result = interp.eval_command(cmd, trace)
        mock_filter.assert_called_once()
        # First argument should be the input trace
        assert np.array_equal(mock_filter.call_args[0][0], trace)


@pytest.mark.unit
class TestEvalPipeline:
    """Test eval_pipeline method."""

    def test_pipeline_single_stage(self, tmp_path: Path) -> None:
        """Test pipeline with single stage."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("data")

        interp = Interpreter()

        mock_load = Mock(return_value={"data": [1, 2, 3]})
        interp.commands["load"] = mock_load
        pipeline = Pipeline(
            line=1,
            column=0,
            stages=[
                Command(
                    line=1,
                    column=0,
                    name="load",
                    args=[Literal(line=1, column=0, value=str(test_file))],
                )
            ],
        )
        result = interp.eval_pipeline(pipeline)
        assert result == {"data": [1, 2, 3]}

    def test_pipeline_multiple_stages(self) -> None:
        """Test pipeline with multiple stages."""
        interp = Interpreter()
        trace = np.array([1.0, 2.0, 3.0])

        mock_load = Mock(return_value=trace)
        mock_filter = Mock(return_value=trace * 0.5)
        interp.commands["load"] = mock_load
        interp.commands["filter"] = mock_filter

        pipeline = Pipeline(
            line=1,
            column=0,
            stages=[
                Command(
                    line=1,
                    column=0,
                    name="load",
                    args=[Literal(line=1, column=0, value="file.csv")],
                ),
                Command(
                    line=1,
                    column=0,
                    name="filter",
                    args=[
                        Literal(line=1, column=0, value="lowpass"),
                        Literal(line=1, column=0, value=1000.0),
                    ],
                ),
            ],
        )

        result = interp.eval_pipeline(pipeline)
        np.testing.assert_array_equal(result, trace * 0.5)

    def test_pipeline_invalid_second_stage(self, tmp_path: Path) -> None:
        """Test pipeline with invalid second stage (non-command)."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("data")

        interp = Interpreter()

        # Mock the load command to avoid actual file loading
        mock_load = Mock(return_value={"data": [1, 2, 3]})
        interp.commands["load"] = mock_load

        # First stage is command (valid), second stage is literal (invalid)
        pipeline = Pipeline(
            line=1,
            column=0,
            stages=[
                Command(
                    line=1,
                    column=0,
                    name="load",
                    args=[Literal(line=1, column=0, value=str(test_file))],
                ),
                Literal(line=1, column=0, value="invalid"),
            ],
        )

        with pytest.raises(InterpreterError, match="Pipeline stage.*must be a command"):
            interp.eval_pipeline(pipeline)


@pytest.mark.unit
class TestEvalStatement:
    """Test eval_statement method."""

    def test_statement_assignment(self) -> None:
        """Test executing assignment statement."""
        interp = Interpreter()
        stmt = Assignment(
            line=1, column=0, variable="my_var", expression=Literal(line=1, column=0, value=42)
        )

        interp.eval_statement(stmt)
        assert interp.variables["my_var"] == 42

    def test_statement_for_loop(self) -> None:
        """Test executing for loop statement."""
        interp = Interpreter()

        with patch.object(interp, "eval_for_loop") as mock_eval:
            mock_eval.return_value = None
            loop = ForLoop(
                line=1,
                column=0,
                variable="i",
                iterable=Literal(line=1, column=0, value=[1, 2, 3]),
                body=[],
            )
            result = interp.eval_statement(loop)
            mock_eval.assert_called_once_with(loop)
            assert result is None

    def test_statement_pipeline(self, tmp_path: Path) -> None:
        """Test executing pipeline statement."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("data")

        interp = Interpreter()

        with patch.object(interp, "eval_pipeline") as mock_eval:
            mock_eval.return_value = {"data": [1, 2, 3]}
            pipeline = Pipeline(
                line=1,
                column=0,
                stages=[
                    Command(
                        line=1,
                        column=0,
                        name="load",
                        args=[Literal(line=1, column=0, value=str(test_file))],
                    )
                ],
            )
            result = interp.eval_statement(pipeline)
            mock_eval.assert_called_once_with(pipeline)
            assert result == {"data": [1, 2, 3]}

    def test_statement_function_call(self) -> None:
        """Test executing function call statement."""
        interp = Interpreter()

        with patch.object(interp, "eval_function_call") as mock_eval:
            mock_eval.return_value = ["file1.csv", "file2.csv"]
            func = FunctionCall(
                line=1, column=0, name="glob", args=[Literal(line=1, column=0, value="*.csv")]
            )
            result = interp.eval_statement(func)
            mock_eval.assert_called_once_with(func)
            assert result == ["file1.csv", "file2.csv"]


@pytest.mark.unit
class TestEvalForLoop:
    """Test eval_for_loop method."""

    def test_for_loop_basic(self) -> None:
        """Test basic for loop execution."""
        interp = Interpreter()
        interp.variables["results"] = []

        # Create a simple for loop that appends to results
        loop_body = [
            Assignment(
                line=2,
                column=0,
                variable="results",
                expression=Literal(line=1, column=0, value=[]),
            )  # Simplified for testing
        ]
        loop = ForLoop(
            line=1,
            column=0,
            variable="i",
            iterable=Literal(line=1, column=0, value=[1, 2, 3]),
            body=loop_body,
        )

        interp.eval_for_loop(loop)

        # Loop variable should be set to last value
        assert interp.variables["i"] == 3

    def test_for_loop_non_iterable(self) -> None:
        """Test for loop with non-iterable."""
        interp = Interpreter()
        loop = ForLoop(
            line=1,
            column=0,
            variable="i",
            iterable=Literal(line=1, column=0, value=42),
            body=[],
        )

        with pytest.raises(InterpreterError, match="not iterable"):
            interp.eval_for_loop(loop)

    def test_for_loop_updates_variable(self) -> None:
        """Test that for loop updates loop variable."""
        interp = Interpreter()
        values = [10, 20, 30]
        loop = ForLoop(
            line=1,
            column=0,
            variable="counter",
            iterable=Literal(line=1, column=0, value=values),
            body=[],
        )

        interp.eval_for_loop(loop)

        assert interp.variables["counter"] == 30  # Last value


@pytest.mark.unit
class TestExecute:
    """Test execute method."""

    def test_execute_empty_program(self) -> None:
        """Test executing empty program."""
        interp = Interpreter()
        interp.execute([])
        assert interp.variables == {}

    def test_execute_assignments(self) -> None:
        """Test executing multiple assignments."""
        interp = Interpreter()
        statements = [
            Assignment(
                line=1, column=0, variable="a", expression=Literal(line=1, column=0, value=10)
            ),
            Assignment(
                line=2, column=0, variable="b", expression=Literal(line=1, column=0, value=20)
            ),
            Assignment(
                line=3, column=0, variable="c", expression=Literal(line=1, column=0, value=30)
            ),
        ]

        interp.execute(statements)
        assert interp.variables["a"] == 10
        assert interp.variables["b"] == 20
        assert interp.variables["c"] == 30


@pytest.mark.unit
class TestExecuteSource:
    """Test execute_source method."""

    def test_execute_source_assignment(self) -> None:
        """Test executing source with assignment."""
        interp = Interpreter()

        with patch("oscura.api.dsl.interpreter.parse_dsl") as mock_parse:
            mock_parse.return_value = [
                Assignment(
                    line=1, column=0, variable="x", expression=Literal(line=1, column=0, value=42)
                )
            ]
            interp.execute_source("$x = 42")
            assert interp.variables["x"] == 42


@pytest.mark.unit
class TestExecuteDSL:
    """Test execute_dsl convenience function."""

    def test_execute_dsl_no_initial_vars(self) -> None:
        """Test execute_dsl without initial variables."""
        with patch("oscura.api.dsl.interpreter.parse_dsl") as mock_parse:
            mock_parse.return_value = [
                Assignment(
                    line=1,
                    column=0,
                    variable="result",
                    expression=Literal(line=1, column=0, value=100),
                )
            ]
            variables = execute_dsl("$result = 100")
            assert variables["result"] == 100

    def test_execute_dsl_with_initial_vars(self) -> None:
        """Test execute_dsl with initial variables."""
        with patch("oscura.api.dsl.interpreter.parse_dsl") as mock_parse:
            mock_parse.return_value = []
            variables = execute_dsl("", variables={"initial": 50})
            assert variables["initial"] == 50

    def test_execute_dsl_returns_final_environment(self) -> None:
        """Test that execute_dsl returns final variable environment."""
        with patch("oscura.api.dsl.interpreter.parse_dsl") as mock_parse:
            mock_parse.return_value = [
                Assignment(
                    line=1, column=0, variable="a", expression=Literal(line=1, column=0, value=10)
                ),
                Assignment(
                    line=2, column=0, variable="b", expression=Literal(line=1, column=0, value=20)
                ),
            ]
            variables = execute_dsl("$a = 10; $b = 20")
            assert variables == {"a": 10, "b": 20}
