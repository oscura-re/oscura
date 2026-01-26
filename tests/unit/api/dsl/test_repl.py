"""Unit tests for DSL REPL module.

Tests for REPL (Read-Eval-Print Loop) functionality including:
- REPL class initialization
- Banner and help display
- Variable listing
- Input reading
- Special command handling (exit, quit, help, vars)
- Line evaluation
- Main REPL loop
- start_repl entry point
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from oscura.api.dsl.interpreter import InterpreterError
from oscura.api.dsl.repl import REPL, start_repl

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestREPLInit:
    """Test REPL initialization."""

    def test_init_creates_interpreter(self) -> None:
        """Test that REPL creates an interpreter instance."""
        repl = REPL()
        assert repl.interpreter is not None
        assert repl.running is True

    def test_init_empty_variables(self) -> None:
        """Test that REPL starts with empty variables."""
        repl = REPL()
        assert repl.interpreter.variables == {}


@pytest.mark.unit
class TestREPLBanner:
    """Test REPL banner display."""

    def test_print_banner(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test printing welcome banner."""
        repl = REPL()
        repl.print_banner()

        captured = capsys.readouterr()
        assert "Oscura DSL REPL" in captured.out
        assert "exit" in captured.out or "quit" in captured.out


@pytest.mark.unit
class TestREPLHelp:
    """Test REPL help display."""

    def test_print_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test printing help message."""
        repl = REPL()
        repl.print_help()

        captured = capsys.readouterr()
        assert "load" in captured.out
        assert "filter" in captured.out
        assert "measure" in captured.out
        assert "plot" in captured.out
        assert "export" in captured.out
        assert "glob" in captured.out
        assert "Variables:" in captured.out
        assert "Pipelines:" in captured.out


@pytest.mark.unit
class TestREPLVariables:
    """Test REPL variable listing."""

    def test_print_variables_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test printing variables when none are defined."""
        repl = REPL()
        repl.print_variables()

        captured = capsys.readouterr()
        assert "No variables defined" in captured.out

    def test_print_variables_with_vars(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test printing defined variables."""
        repl = REPL()
        repl.interpreter.variables["test_var"] = 42
        repl.interpreter.variables["another_var"] = "hello"

        repl.print_variables()

        captured = capsys.readouterr()
        assert "test_var" in captured.out
        assert "42" in captured.out
        assert "another_var" in captured.out
        assert "hello" in captured.out

    def test_print_variables_truncates_long_values(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that long variable values are truncated."""
        repl = REPL()
        long_value = "x" * 100
        repl.interpreter.variables["long_var"] = long_value

        repl.print_variables()

        captured = capsys.readouterr()
        assert "long_var" in captured.out
        assert "..." in captured.out  # Truncation indicator
        assert len(captured.out) < len(long_value) + 50  # Should be truncated


@pytest.mark.unit
class TestREPLReadInput:
    """Test REPL input reading."""

    def test_read_input_normal(self) -> None:
        """Test reading normal input."""
        repl = REPL()

        with patch("builtins.input", return_value="test input"):
            result = repl.read_input()
            assert result == "test input"

    def test_read_input_eof(self) -> None:
        """Test reading input with EOF."""
        repl = REPL()

        with patch("builtins.input", side_effect=EOFError):
            result = repl.read_input()
            assert result is None


@pytest.mark.unit
class TestREPLSpecialCommands:
    """Test REPL special command handling."""

    def test_special_command_exit(self) -> None:
        """Test 'exit' special command."""
        repl = REPL()
        assert repl.running is True

        result = repl.eval_special_command("exit")

        assert result is True
        assert repl.running is False

    def test_special_command_quit(self) -> None:
        """Test 'quit' special command."""
        repl = REPL()
        assert repl.running is True

        result = repl.eval_special_command("quit")

        assert result is True
        assert repl.running is False

    def test_special_command_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test 'help' special command."""
        repl = REPL()

        result = repl.eval_special_command("help")

        assert result is True
        captured = capsys.readouterr()
        assert "Commands:" in captured.out

    def test_special_command_vars(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test 'vars' special command."""
        repl = REPL()
        repl.interpreter.variables["test"] = 123

        result = repl.eval_special_command("vars")

        assert result is True
        captured = capsys.readouterr()
        assert "test" in captured.out

    def test_special_command_not_special(self) -> None:
        """Test non-special command returns False."""
        repl = REPL()

        result = repl.eval_special_command("load file.csv")

        assert result is False

    def test_special_command_with_whitespace(self) -> None:
        """Test special command with leading/trailing whitespace."""
        repl = REPL()

        result = repl.eval_special_command("  exit  ")

        assert result is True
        assert repl.running is False


@pytest.mark.unit
class TestREPLEvalLine:
    """Test REPL line evaluation."""

    def test_eval_line_empty(self) -> None:
        """Test evaluating empty line."""
        repl = REPL()

        # Should not raise any errors
        repl.eval_line("")
        repl.eval_line("   ")

        # Verify no variables created from empty lines
        assert len(repl.interpreter.variables) == 0

    def test_eval_line_comment(self) -> None:
        """Test evaluating comment line."""
        repl = REPL()

        # Comments should be ignored
        repl.eval_line("# This is a comment")

        # No variables should be defined
        assert len(repl.interpreter.variables) == 0

    def test_eval_line_special_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test evaluating special command."""
        repl = REPL()

        repl.eval_line("help")

        captured = capsys.readouterr()
        assert "Commands:" in captured.out

    def test_eval_line_assignment(self) -> None:
        """Test evaluating assignment."""
        repl = REPL()

        with patch("oscura.api.dsl.repl.parse_dsl") as mock_parse:
            from oscura.api.dsl.parser import Assignment, Literal

            mock_parse.return_value = [
                Assignment(
                    line=1, column=0, variable="x", expression=Literal(line=1, column=0, value=42)
                )
            ]

            repl.eval_line("$x = 42")

            assert repl.interpreter.variables["x"] == 42

    def test_eval_line_syntax_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test handling syntax errors."""
        repl = REPL()

        with patch("oscura.api.dsl.repl.parse_dsl", side_effect=SyntaxError("Invalid syntax")):
            repl.eval_line("invalid syntax")

        captured = capsys.readouterr()
        assert "Syntax error" in captured.err

    def test_eval_line_interpreter_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test handling interpreter errors."""
        repl = REPL()

        with patch(
            "oscura.api.dsl.repl.parse_dsl", side_effect=InterpreterError("Execution error")
        ):
            repl.eval_line("load nonexistent.csv")

        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_eval_line_unexpected_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test handling unexpected errors."""
        repl = REPL()

        with patch("oscura.api.dsl.repl.parse_dsl", side_effect=RuntimeError("Unexpected")):
            repl.eval_line("something")

        captured = capsys.readouterr()
        assert "Unexpected error" in captured.err


@pytest.mark.unit
class TestREPLRun:
    """Test REPL main loop."""

    def test_run_single_command_then_exit(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test running REPL with single command then exit."""
        repl = REPL()

        with patch.object(repl, "read_input", side_effect=["help", "exit"]):
            repl.run()

        captured = capsys.readouterr()
        assert "Oscura DSL REPL" in captured.out
        assert "Goodbye!" in captured.out

    def test_run_eof_exit(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test running REPL with EOF exit."""
        repl = REPL()

        with patch.object(repl, "read_input", return_value=None):
            repl.run()

        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    def test_run_executes_commands(self) -> None:
        """Test that run executes commands."""
        repl = REPL()

        with patch.object(repl, "read_input", side_effect=["vars", None]):
            with patch.object(repl, "eval_line") as mock_eval:
                repl.run()
                mock_eval.assert_called_once_with("vars")
                # Verify eval_line was called with correct argument
                assert mock_eval.call_count == 1

    def test_run_handles_multiple_lines(self) -> None:
        """Test running REPL with multiple input lines."""
        repl = REPL()

        # Add None at the end to signal EOF and prevent StopIteration
        inputs = ["# comment", "", "help", "vars", "exit", None]

        with patch.object(repl, "read_input", side_effect=inputs):
            with patch.object(repl, "eval_line") as mock_eval:
                repl.run()
                # Should process all inputs except the last None (which triggers EOF)
                assert mock_eval.call_count == len(inputs) - 1


@pytest.mark.unit
class TestStartREPL:
    """Test start_repl entry point."""

    def test_start_repl_creates_and_runs(self) -> None:
        """Test that start_repl creates REPL instance and runs it."""
        with patch("oscura.api.dsl.repl.REPL") as mock_repl_class:
            mock_instance = Mock()
            mock_repl_class.return_value = mock_instance

            start_repl()

            mock_repl_class.assert_called_once()
            mock_instance.run.assert_called_once()
            # Verify proper creation and execution flow
            assert mock_repl_class.call_count == 1
            assert mock_instance.run.call_count == 1


@pytest.mark.unit
class TestREPLIntegration:
    """Integration tests for REPL functionality."""

    def test_repl_assignment_and_retrieval(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test assigning and retrieving variables."""
        repl = REPL()

        with patch("oscura.api.dsl.repl.parse_dsl") as mock_parse:
            from oscura.api.dsl.parser import Assignment, Literal

            # First assignment
            mock_parse.return_value = [
                Assignment(
                    line=1,
                    column=0,
                    variable="my_var",
                    expression=Literal(line=1, column=0, value=100),
                )
            ]
            repl.eval_line("$my_var = 100")

            # Check variable was set
            repl.eval_line("vars")

        captured = capsys.readouterr()
        assert "my_var" in captured.out
        assert "100" in captured.out

    def test_repl_preserves_state_across_lines(self) -> None:
        """Test that REPL preserves state across multiple line evaluations."""
        repl = REPL()

        with patch("oscura.api.dsl.repl.parse_dsl") as mock_parse:
            from oscura.api.dsl.parser import Assignment, Literal

            # First line
            mock_parse.return_value = [
                Assignment(
                    line=1, column=0, variable="a", expression=Literal(line=1, column=0, value=10)
                )
            ]
            repl.eval_line("$a = 10")

            # Second line
            mock_parse.return_value = [
                Assignment(
                    line=1, column=0, variable="b", expression=Literal(line=1, column=0, value=20)
                )
            ]
            repl.eval_line("$b = 20")

        # Both variables should be preserved
        assert repl.interpreter.variables["a"] == 10
        assert repl.interpreter.variables["b"] == 20
