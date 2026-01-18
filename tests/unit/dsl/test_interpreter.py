"""Tests for DSL interpreter.

Requirements tested:
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from oscura.dsl.interpreter import (
    Interpreter,
    InterpreterError,
    execute_dsl,
)
from oscura.dsl.parser import (
    Assignment,
    Command,
    ForLoop,
    FunctionCall,
    Literal,
    Pipeline,
    Variable,
)

pytestmark = pytest.mark.unit


class TestInterpreterError:
    """Test InterpreterError exception class."""

    def test_interpreter_error_is_exception(self):
        """Test that InterpreterError is an Exception."""
        error = InterpreterError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"

    def test_interpreter_error_message(self):
        """Test InterpreterError with custom message."""
        msg = "Custom interpreter error"
        error = InterpreterError(msg)
        assert str(error) == msg


class TestInterpreterInit:
    """Test Interpreter initialization."""

    def test_init_empty_environment(self):
        """Test interpreter initializes with empty environment."""
        interp = Interpreter()
        assert interp.variables == {}
        assert isinstance(interp.commands, dict)
        assert len(interp.commands) > 0  # Should have built-in commands

    def test_init_registers_builtin_commands(self):
        """Test that built-in commands are registered on init."""
        interp = Interpreter()
        expected_commands = ["load", "filter", "measure", "plot", "export", "glob"]
        for cmd in expected_commands:
            assert cmd in interp.commands
            assert callable(interp.commands[cmd])


class TestBuiltinCommands:
    """Test built-in command implementations."""

    def test_cmd_load_wrong_arg_count(self):
        """Test load command with wrong number of arguments."""
        interp = Interpreter()

        with pytest.raises(InterpreterError, match="load requires exactly 1 argument"):
            interp._cmd_load()

        with pytest.raises(InterpreterError, match="load requires exactly 1 argument"):
            interp._cmd_load("file1.csv", "file2.csv")

    def test_cmd_load_non_string_argument(self):
        """Test load command with non-string argument."""
        interp = Interpreter()

        with pytest.raises(InterpreterError, match="filename must be a string"):
            interp._cmd_load(123)

    def test_cmd_load_file_not_found(self):
        """Test load command with non-existent file."""
        interp = Interpreter()

        with pytest.raises(InterpreterError, match="File not found"):
            interp._cmd_load("/nonexistent/file.csv")

    def test_cmd_filter_no_args(self):
        """Test filter command with no arguments."""
        interp = Interpreter()

        with pytest.raises(InterpreterError, match="filter requires filter type"):
            interp._cmd_filter("data")

    def test_cmd_filter_non_string_type(self):
        """Test filter command with non-string filter type."""
        interp = Interpreter()

        with pytest.raises(InterpreterError, match="filter type must be a string"):
            interp._cmd_filter("data", 123)

    def test_cmd_measure_no_args(self):
        """Test measure command with no arguments."""
        interp = Interpreter()

        with pytest.raises(InterpreterError, match="measure requires measurement name"):
            interp._cmd_measure("data")

    def test_cmd_measure_non_string_name(self):
        """Test measure command with non-string measurement name."""
        interp = Interpreter()

        with pytest.raises(InterpreterError, match="measurement name must be a string"):
            interp._cmd_measure("data", 123)

    def test_cmd_export_no_args(self):
        """Test export command with no arguments."""
        interp = Interpreter()

        with pytest.raises(InterpreterError, match="export requires format argument"):
            interp._cmd_export("data")

    def test_cmd_export_non_string_format(self):
        """Test export command with non-string format."""
        interp = Interpreter()

        with pytest.raises(InterpreterError, match="export format must be a string"):
            interp._cmd_export("data", 123)

    def test_cmd_glob_non_string_pattern(self):
        """Test glob command with non-string pattern."""
        interp = Interpreter()

        with pytest.raises(InterpreterError, match="glob pattern must be a string"):
            interp._cmd_glob(123)

    def test_cmd_glob_valid_pattern(self):
        """Test glob command with valid pattern."""
        interp = Interpreter()

        # Create temporary directory with test files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "test1.txt").touch()
            (tmp_path / "test2.txt").touch()
            (tmp_path / "other.csv").touch()

            pattern = str(tmp_path / "*.txt")
            result = interp._cmd_glob(pattern)

            assert isinstance(result, list)
            assert len(result) == 2
            assert all(p.endswith(".txt") for p in result)

    def test_cmd_glob_no_matches(self):
        """Test glob command with pattern that matches nothing."""
        interp = Interpreter()
        result = interp._cmd_glob("/nonexistent/path/*.xyz")
        assert result == []


class TestEvalExpression:
    """Test expression evaluation."""

    def test_eval_literal_string(self):
        """Test evaluating string literal."""
        interp = Interpreter()
        expr = Literal(value="hello", line=1, column=1)
        result = interp.eval_expression(expr)
        assert result == "hello"

    def test_eval_literal_number(self):
        """Test evaluating number literal."""
        interp = Interpreter()
        expr = Literal(value=42, line=1, column=1)
        result = interp.eval_expression(expr)
        assert result == 42

    def test_eval_literal_float(self):
        """Test evaluating float literal."""
        interp = Interpreter()
        expr = Literal(value=3.14, line=1, column=1)
        result = interp.eval_expression(expr)
        assert result == 3.14

    def test_eval_variable_defined(self):
        """Test evaluating defined variable."""
        interp = Interpreter()
        interp.variables["$data"] = "test_value"
        expr = Variable(name="$data", line=1, column=1)
        result = interp.eval_expression(expr)
        assert result == "test_value"

    def test_eval_variable_undefined(self):
        """Test evaluating undefined variable raises error."""
        interp = Interpreter()
        expr = Variable(name="$undefined", line=5, column=3)

        with pytest.raises(InterpreterError, match="Undefined variable: \\$undefined at line 5"):
            interp.eval_expression(expr)

    def test_eval_function_call(self):
        """Test evaluating function call."""
        interp = Interpreter()

        # Mock a command function
        mock_cmd = Mock(return_value="result")
        interp.commands["test_func"] = mock_cmd

        expr = FunctionCall(
            name="test_func",
            args=[Literal(value="arg1", line=1, column=1)],
            line=1,
            column=1,
        )
        result = interp.eval_expression(expr)

        assert result == "result"
        mock_cmd.assert_called_once_with("arg1")

    def test_eval_command(self):
        """Test evaluating command expression."""
        interp = Interpreter()

        # Mock a command function
        mock_cmd = Mock(return_value="result")
        interp.commands["test_cmd"] = mock_cmd

        expr = Command(
            name="test_cmd",
            args=[Literal(value="arg1", line=1, column=1)],
            line=1,
            column=1,
        )
        result = interp.eval_expression(expr)

        assert result == "result"
        mock_cmd.assert_called_once_with("arg1")

    def test_eval_pipeline(self):
        """Test evaluating pipeline expression."""
        interp = Interpreter()

        # Create a simple pipeline: "value" | cmd1 | cmd2
        mock_cmd1 = Mock(return_value="intermediate")
        mock_cmd2 = Mock(return_value="final")
        interp.commands["cmd1"] = mock_cmd1
        interp.commands["cmd2"] = mock_cmd2

        pipeline = Pipeline(
            stages=[
                Literal(value="value", line=1, column=1),
                Command(name="cmd1", args=[], line=1, column=10),
                Command(name="cmd2", args=[], line=1, column=17),
            ],
            line=1,
            column=1,
        )

        result = interp.eval_expression(pipeline)
        assert result == "final"

    def test_eval_unknown_expression_type(self):
        """Test evaluating unknown expression type raises error."""
        interp = Interpreter()

        # Create a mock expression with unknown type
        class UnknownExpr:
            pass

        expr = UnknownExpr()
        with pytest.raises(InterpreterError, match="Unknown expression type: UnknownExpr"):
            interp.eval_expression(expr)


class TestEvalFunctionCall:
    """Test function call evaluation."""

    def test_eval_function_call_unknown_function(self):
        """Test calling unknown function raises error."""
        interp = Interpreter()
        func = FunctionCall(name="unknown_func", args=[], line=3, column=5)

        with pytest.raises(InterpreterError, match="Unknown function: unknown_func at line 3"):
            interp.eval_function_call(func)

    def test_eval_function_call_with_args(self):
        """Test function call with multiple arguments."""
        interp = Interpreter()

        mock_func = Mock(return_value="result")
        interp.commands["func"] = mock_func

        func = FunctionCall(
            name="func",
            args=[
                Literal(value="arg1", line=1, column=1),
                Literal(value=42, line=1, column=8),
            ],
            line=1,
            column=1,
        )

        result = interp.eval_function_call(func)
        assert result == "result"
        mock_func.assert_called_once_with("arg1", 42)

    def test_eval_function_call_with_variable_args(self):
        """Test function call with variable arguments."""
        interp = Interpreter()
        interp.variables["$var"] = "var_value"

        mock_func = Mock(return_value="result")
        interp.commands["func"] = mock_func

        func = FunctionCall(
            name="func",
            args=[Variable(name="$var", line=1, column=1)],
            line=1,
            column=1,
        )

        result = interp.eval_function_call(func)
        assert result == "result"
        mock_func.assert_called_once_with("var_value")


class TestEvalCommand:
    """Test command evaluation."""

    def test_eval_command_unknown_command(self):
        """Test evaluating unknown command raises error."""
        interp = Interpreter()
        cmd = Command(name="unknown_cmd", args=[], line=7, column=2)

        with pytest.raises(InterpreterError, match="Unknown command: unknown_cmd at line 7"):
            interp.eval_command(cmd, None)

    def test_eval_command_no_input(self):
        """Test command with no piped input."""
        interp = Interpreter()

        mock_cmd = Mock(return_value="result")
        interp.commands["cmd"] = mock_cmd

        cmd = Command(
            name="cmd",
            args=[Literal(value="arg", line=1, column=1)],
            line=1,
            column=1,
        )

        result = interp.eval_command(cmd, None)
        assert result == "result"
        mock_cmd.assert_called_once_with("arg")

    def test_eval_command_with_input(self):
        """Test command with piped input."""
        interp = Interpreter()

        mock_cmd = Mock(return_value="result")
        interp.commands["cmd"] = mock_cmd

        cmd = Command(
            name="cmd",
            args=[Literal(value="arg", line=1, column=1)],
            line=1,
            column=1,
        )

        result = interp.eval_command(cmd, "input_data")
        assert result == "result"
        # Input data should be prepended to arguments
        mock_cmd.assert_called_once_with("input_data", "arg")

    def test_eval_command_evaluates_args(self):
        """Test that command arguments are evaluated."""
        interp = Interpreter()
        interp.variables["$var"] = "evaluated"

        mock_cmd = Mock(return_value="result")
        interp.commands["cmd"] = mock_cmd

        cmd = Command(
            name="cmd",
            args=[Variable(name="$var", line=1, column=1)],
            line=1,
            column=1,
        )

        interp.eval_command(cmd, None)
        mock_cmd.assert_called_once_with("evaluated")


class TestEvalPipeline:
    """Test pipeline evaluation."""

    def test_eval_pipeline_single_stage(self):
        """Test pipeline with single stage."""
        interp = Interpreter()

        pipeline = Pipeline(
            stages=[Literal(value="value", line=1, column=1)],
            line=1,
            column=1,
        )

        result = interp.eval_pipeline(pipeline)
        assert result == "value"

    def test_eval_pipeline_multiple_stages(self):
        """Test pipeline with multiple stages."""
        interp = Interpreter()

        mock_cmd1 = Mock(return_value="stage1_result")
        mock_cmd2 = Mock(return_value="stage2_result")
        interp.commands["cmd1"] = mock_cmd1
        interp.commands["cmd2"] = mock_cmd2

        pipeline = Pipeline(
            stages=[
                Literal(value="initial", line=1, column=1),
                Command(name="cmd1", args=[], line=1, column=10),
                Command(name="cmd2", args=[], line=1, column=17),
            ],
            line=1,
            column=1,
        )

        result = interp.eval_pipeline(pipeline)
        assert result == "stage2_result"

        # First command gets initial value
        mock_cmd1.assert_called_once_with("initial")
        # Second command gets result from first
        mock_cmd2.assert_called_once_with("stage1_result")

    def test_eval_pipeline_first_stage_command(self):
        """Test pipeline starting with a command."""
        interp = Interpreter()

        mock_cmd1 = Mock(return_value="cmd1_result")
        mock_cmd2 = Mock(return_value="cmd2_result")
        interp.commands["cmd1"] = mock_cmd1
        interp.commands["cmd2"] = mock_cmd2

        pipeline = Pipeline(
            stages=[
                Command(name="cmd1", args=[], line=1, column=1),
                Command(name="cmd2", args=[], line=1, column=7),
            ],
            line=1,
            column=1,
        )

        result = interp.eval_pipeline(pipeline)
        assert result == "cmd2_result"

        # First command gets no input
        mock_cmd1.assert_called_once_with()
        # Second command gets result from first
        mock_cmd2.assert_called_once_with("cmd1_result")

    def test_eval_pipeline_non_command_stage(self):
        """Test pipeline with non-command in later stage raises error."""
        interp = Interpreter()

        pipeline = Pipeline(
            stages=[
                Literal(value="initial", line=1, column=1),
                Literal(value="invalid", line=1, column=10),  # Can't pipe to literal
            ],
            line=1,
            column=1,
        )

        with pytest.raises(InterpreterError, match="Pipeline stage 2 must be a command"):
            interp.eval_pipeline(pipeline)


class TestEvalStatement:
    """Test statement evaluation."""

    def test_eval_assignment(self):
        """Test assignment statement."""
        interp = Interpreter()

        stmt = Assignment(
            variable="$data",
            expression=Literal(value="test_value", line=1, column=1),
            line=1,
            column=1,
        )

        interp.eval_statement(stmt)
        assert interp.variables["$data"] == "test_value"

    def test_eval_assignment_overwrite(self):
        """Test assignment overwrites existing variable."""
        interp = Interpreter()
        interp.variables["$data"] = "old_value"

        stmt = Assignment(
            variable="$data",
            expression=Literal(value="new_value", line=1, column=1),
            line=1,
            column=1,
        )

        interp.eval_statement(stmt)
        assert interp.variables["$data"] == "new_value"

    def test_eval_for_loop(self):
        """Test for loop statement."""
        interp = Interpreter()

        # Track execution
        executed = []

        def mock_cmd(item):
            executed.append(item)
            return None

        interp.commands["track"] = mock_cmd

        # Commands in for loop body should be wrapped in Pipeline
        loop = ForLoop(
            variable="$item",
            iterable=Literal(value=[1, 2, 3], line=1, column=1),
            body=[
                Pipeline(
                    stages=[
                        Command(
                            name="track",
                            args=[Variable(name="$item", line=2, column=5)],
                            line=2,
                            column=1,
                        )
                    ],
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )

        interp.eval_statement(loop)
        assert executed == [1, 2, 3]

    def test_eval_pipeline_statement(self):
        """Test pipeline as statement."""
        interp = Interpreter()

        mock_cmd = Mock(return_value="result")
        interp.commands["cmd"] = mock_cmd

        pipeline = Pipeline(
            stages=[Command(name="cmd", args=[], line=1, column=1)],
            line=1,
            column=1,
        )

        interp.eval_statement(pipeline)
        mock_cmd.assert_called_once()

    def test_eval_unknown_statement_type(self):
        """Test unknown statement type raises error."""
        interp = Interpreter()

        class UnknownStmt:
            pass

        stmt = UnknownStmt()
        with pytest.raises(InterpreterError, match="Unknown statement type: UnknownStmt"):
            interp.eval_statement(stmt)


class TestEvalForLoop:
    """Test for loop evaluation."""

    def test_eval_for_loop_empty_iterable(self):
        """Test for loop with empty iterable."""
        interp = Interpreter()

        executed = []

        def mock_cmd(item):
            executed.append(item)

        interp.commands["track"] = mock_cmd

        loop = ForLoop(
            variable="$item",
            iterable=Literal(value=[], line=1, column=1),
            body=[
                Pipeline(
                    stages=[
                        Command(
                            name="track",
                            args=[Variable(name="$item", line=2, column=1)],
                            line=2,
                            column=1,
                        )
                    ],
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )

        interp.eval_for_loop(loop)
        assert executed == []

    def test_eval_for_loop_multiple_statements(self):
        """Test for loop with multiple statements in body."""
        interp = Interpreter()

        executed = []

        def track1(item):
            executed.append(("track1", item))

        def track2(item):
            executed.append(("track2", item))

        interp.commands["track1"] = track1
        interp.commands["track2"] = track2

        loop = ForLoop(
            variable="$item",
            iterable=Literal(value=["a", "b"], line=1, column=1),
            body=[
                Pipeline(
                    stages=[
                        Command(
                            name="track1",
                            args=[Variable(name="$item", line=2, column=1)],
                            line=2,
                            column=1,
                        )
                    ],
                    line=2,
                    column=1,
                ),
                Pipeline(
                    stages=[
                        Command(
                            name="track2",
                            args=[Variable(name="$item", line=3, column=1)],
                            line=3,
                            column=1,
                        )
                    ],
                    line=3,
                    column=1,
                ),
            ],
            line=1,
            column=1,
        )

        interp.eval_for_loop(loop)
        assert executed == [
            ("track1", "a"),
            ("track2", "a"),
            ("track1", "b"),
            ("track2", "b"),
        ]

    def test_eval_for_loop_non_iterable(self):
        """Test for loop with non-iterable raises error."""
        interp = Interpreter()

        loop = ForLoop(
            variable="$item",
            iterable=Literal(value=42, line=1, column=1),  # Not iterable
            body=[],
            line=1,
            column=1,
        )

        with pytest.raises(InterpreterError, match="not iterable.*int at line 1"):
            interp.eval_for_loop(loop)

    def test_eval_for_loop_variable_scope(self):
        """Test for loop sets loop variable correctly."""
        interp = Interpreter()

        loop = ForLoop(
            variable="$i",
            iterable=Literal(value=[10, 20], line=1, column=1),
            body=[],
            line=1,
            column=1,
        )

        interp.eval_for_loop(loop)
        # Variable should be set to last item
        assert interp.variables["$i"] == 20

    def test_eval_for_loop_with_function_call_iterable(self):
        """Test for loop with function call as iterable."""
        interp = Interpreter()

        executed = []

        def track(item):
            executed.append(item)

        interp.commands["track"] = track
        interp.commands["get_items"] = lambda: ["x", "y", "z"]

        loop = ForLoop(
            variable="$item",
            iterable=FunctionCall(name="get_items", args=[], line=1, column=1),
            body=[
                Pipeline(
                    stages=[
                        Command(
                            name="track",
                            args=[Variable(name="$item", line=2, column=1)],
                            line=2,
                            column=1,
                        )
                    ],
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )

        interp.eval_for_loop(loop)
        assert executed == ["x", "y", "z"]


class TestExecute:
    """Test program execution."""

    def test_execute_empty_program(self):
        """Test executing empty program."""
        interp = Interpreter()
        interp.execute([])
        assert interp.variables == {}

    def test_execute_single_statement(self):
        """Test executing single statement."""
        interp = Interpreter()

        stmt = Assignment(
            variable="$x",
            expression=Literal(value=42, line=1, column=1),
            line=1,
            column=1,
        )

        interp.execute([stmt])
        assert interp.variables["$x"] == 42

    def test_execute_multiple_statements(self):
        """Test executing multiple statements."""
        interp = Interpreter()

        stmts = [
            Assignment(
                variable="$x",
                expression=Literal(value=10, line=1, column=1),
                line=1,
                column=1,
            ),
            Assignment(
                variable="$y",
                expression=Literal(value=20, line=2, column=1),
                line=2,
                column=1,
            ),
        ]

        interp.execute(stmts)
        assert interp.variables["$x"] == 10
        assert interp.variables["$y"] == 20

    def test_execute_statements_in_order(self):
        """Test statements are executed in order."""
        interp = Interpreter()

        stmts = [
            Assignment(
                variable="$x",
                expression=Literal(value=10, line=1, column=1),
                line=1,
                column=1,
            ),
            Assignment(
                variable="$y",
                expression=Variable(name="$x", line=2, column=1),
                line=2,
                column=1,
            ),
        ]

        interp.execute(stmts)
        assert interp.variables["$y"] == 10


class TestExecuteSource:
    """Test source code execution."""

    def test_execute_source_simple_assignment(self):
        """Test executing simple assignment from source."""
        interp = Interpreter()
        interp.execute_source('$x = "hello"')
        assert interp.variables["$x"] == "hello"

    def test_execute_source_multiple_lines(self):
        """Test executing multiple lines from source."""
        interp = Interpreter()
        source = "$x = 10\n$y = 20"
        interp.execute_source(source)
        assert interp.variables["$x"] == 10
        assert interp.variables["$y"] == 20

    def test_execute_source_syntax_error(self):
        """Test executing invalid source raises SyntaxError."""
        interp = Interpreter()

        with pytest.raises(SyntaxError):
            interp.execute_source("$x = ")

    def test_execute_source_runtime_error(self):
        """Test executing source with runtime error."""
        interp = Interpreter()

        # Reference undefined variable
        with pytest.raises(InterpreterError, match="Undefined variable"):
            interp.execute_source("$x = $undefined")


class TestExecuteDSL:
    """Test execute_dsl convenience function."""

    def test_execute_dsl_simple(self):
        """Test execute_dsl with simple code."""
        result = execute_dsl("$x = 42")
        assert result["$x"] == 42

    def test_execute_dsl_returns_variables(self):
        """Test execute_dsl returns variable environment."""
        result = execute_dsl("$a = 1\n$b = 2")
        assert result == {"$a": 1, "$b": 2}

    def test_execute_dsl_with_initial_variables(self):
        """Test execute_dsl with initial variables."""
        initial = {"$x": 10}
        result = execute_dsl("$y = $x", variables=initial)
        assert result["$x"] == 10
        assert result["$y"] == 10

    def test_execute_dsl_overwrites_initial_variables(self):
        """Test execute_dsl can overwrite initial variables."""
        initial = {"$x": 10}
        result = execute_dsl("$x = 20", variables=initial)
        assert result["$x"] == 20

    def test_execute_dsl_empty_source(self):
        """Test execute_dsl with empty source."""
        result = execute_dsl("")
        assert result == {}

    def test_execute_dsl_syntax_error(self):
        """Test execute_dsl with syntax error."""
        with pytest.raises(SyntaxError):
            execute_dsl("invalid syntax )")

    def test_execute_dsl_runtime_error(self):
        """Test execute_dsl with runtime error."""
        with pytest.raises(InterpreterError):
            execute_dsl("$x = $undefined")


class TestDslInterpreterIntegration:
    """Integration tests with parser."""

    def test_integration_assignment_and_reference(self):
        """Test assignment followed by variable reference."""
        source = '$x = "value"\n$y = $x'
        result = execute_dsl(source)
        assert result["$x"] == "value"
        assert result["$y"] == "value"

    def test_integration_glob_command(self):
        """Test glob command integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "file1.txt").touch()
            (tmp_path / "file2.txt").touch()

            pattern = str(tmp_path / "*.txt")
            source = f'$files = glob("{pattern}")'
            result = execute_dsl(source)

            assert isinstance(result["$files"], list)
            assert len(result["$files"]) == 2

    def test_integration_for_loop_with_glob(self):
        """Test for loop with glob integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "a.txt").touch()
            (tmp_path / "b.txt").touch()

            # We can't fully test load inside loop due to not implemented,
            # but we can test the loop structure
            interp = Interpreter()

            # Track what files are processed
            processed = []

            def mock_process(filename):
                processed.append(filename)
                return None

            interp.commands["process"] = mock_process

            pattern = str(tmp_path / "*.txt")
            # Use proper DSL syntax - for loops with single command work via parser
            source = f'for $f in glob("{pattern}"): process($f)'
            interp.execute_source(source)

            assert len(processed) == 2
            assert all(f.endswith(".txt") for f in processed)

    def test_integration_complex_program(self):
        """Test complex program with multiple features."""
        source = """
$x = 10
$y = 20
$z = $x
"""
        result = execute_dsl(source)
        assert result["$x"] == 10
        assert result["$y"] == 20
        assert result["$z"] == 10

    def test_integration_literal_types(self):
        """Test various literal types in assignments."""
        source = """
$str = "hello"
$int = 42
$float = 3.14
"""
        result = execute_dsl(source)
        assert result["$str"] == "hello"
        assert result["$int"] == 42
        assert result["$float"] == 3.14

    def test_integration_nested_for_loops(self):
        """Test nested for loops."""
        interp = Interpreter()

        executed = []

        def track(i, j):
            executed.append((i, j))

        interp.commands["track"] = track

        # Parse nested loops - Commands must be wrapped in Pipeline for statements
        outer_loop = ForLoop(
            variable="$i",
            iterable=Literal(value=[1, 2], line=1, column=1),
            body=[
                ForLoop(
                    variable="$j",
                    iterable=Literal(value=["a", "b"], line=2, column=1),
                    body=[
                        Pipeline(
                            stages=[
                                Command(
                                    name="track",
                                    args=[
                                        Variable(name="$i", line=3, column=1),
                                        Variable(name="$j", line=3, column=4),
                                    ],
                                    line=3,
                                    column=1,
                                )
                            ],
                            line=3,
                            column=1,
                        )
                    ],
                    line=2,
                    column=1,
                )
            ],
            line=1,
            column=1,
        )

        interp.eval_for_loop(outer_loop)
        assert executed == [(1, "a"), (1, "b"), (2, "a"), (2, "b")]
