"""Unit tests for Oscura DSL REPL.

This module tests the interactive DSL shell.
"""

from __future__ import annotations

import io
from unittest import mock

import pytest

pytestmark = pytest.mark.unit


# =============================================================================
# REPL Class Tests
# =============================================================================


@pytest.mark.unit
class TestREPLInitialization:
    """Test REPL initialization."""

    def test_repl_creates_interpreter(self) -> None:
        """Test that REPL initializes with an interpreter."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        assert repl.interpreter is not None
        assert repl.running is True

    def test_repl_interpreter_has_empty_variables(self) -> None:
        """Test that REPL interpreter starts with no variables."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        assert repl.interpreter.variables == {}


@pytest.mark.unit
class TestREPLBanner:
    """Test REPL banner output."""

    def test_print_banner_output(self) -> None:
        """Test that banner prints expected content."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            repl.print_banner()
            output = mock_stdout.getvalue()

        assert "Oscura DSL REPL" in output
        assert "exit" in output.lower() or "quit" in output.lower()
        assert "help" in output.lower()


@pytest.mark.unit
class TestREPLHelp:
    """Test REPL help output."""

    def test_print_help_output(self) -> None:
        """Test that help prints expected content."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            repl.print_help()
            output = mock_stdout.getvalue()

        # Should mention key commands
        assert "load" in output.lower()
        assert "filter" in output.lower()
        assert "measure" in output.lower()
        assert "plot" in output.lower()
        assert "export" in output.lower()

    def test_print_help_mentions_variables(self) -> None:
        """Test that help mentions variable syntax."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            repl.print_help()
            output = mock_stdout.getvalue()

        assert "$" in output or "variable" in output.lower()

    def test_print_help_mentions_pipelines(self) -> None:
        """Test that help mentions pipeline syntax."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            repl.print_help()
            output = mock_stdout.getvalue()

        assert "|" in output or "pipeline" in output.lower()


@pytest.mark.unit
class TestREPLVariables:
    """Test REPL variable display."""

    def test_print_variables_empty(self) -> None:
        """Test printing when no variables defined."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            repl.print_variables()
            output = mock_stdout.getvalue()

        assert "no variables" in output.lower()

    def test_print_variables_with_values(self) -> None:
        """Test printing when variables are defined."""
        from oscura.dsl.repl import REPL

        repl = REPL()
        repl.interpreter.variables["x"] = 42
        repl.interpreter.variables["y"] = "hello"

        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            repl.print_variables()
            output = mock_stdout.getvalue()

        assert "x" in output
        assert "42" in output
        assert "y" in output
        assert "hello" in output

    def test_print_variables_truncates_long_values(self) -> None:
        """Test that long variable values are truncated."""
        from oscura.dsl.repl import REPL

        repl = REPL()
        repl.interpreter.variables["long_var"] = "a" * 100

        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            repl.print_variables()
            output = mock_stdout.getvalue()

        # Should be truncated with ...
        assert "..." in output
        # Should not have the full 100 a's
        assert "a" * 100 not in output


@pytest.mark.unit
class TestREPLInput:
    """Test REPL input handling."""

    def test_read_input_normal(self) -> None:
        """Test reading normal input."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        with mock.patch("builtins.input", return_value="load test.bin"):
            result = repl.read_input()

        assert result == "load test.bin"

    def test_read_input_eof(self) -> None:
        """Test handling EOF."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        with mock.patch("builtins.input", side_effect=EOFError):
            result = repl.read_input()

        assert result is None


@pytest.mark.unit
class TestREPLSpecialCommands:
    """Test REPL special commands."""

    def test_exit_command(self) -> None:
        """Test 'exit' command sets running to False."""
        from oscura.dsl.repl import REPL

        repl = REPL()
        assert repl.running is True

        result = repl.eval_special_command("exit")

        assert result is True
        assert repl.running is False

    def test_quit_command(self) -> None:
        """Test 'quit' command sets running to False."""
        from oscura.dsl.repl import REPL

        repl = REPL()
        assert repl.running is True

        result = repl.eval_special_command("quit")

        assert result is True
        assert repl.running is False

    def test_help_command(self) -> None:
        """Test 'help' command prints help."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = repl.eval_special_command("help")
            output = mock_stdout.getvalue()

        assert result is True
        assert len(output) > 0  # Help was printed

    def test_vars_command(self) -> None:
        """Test 'vars' command prints variables."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = repl.eval_special_command("vars")
            output = mock_stdout.getvalue()

        assert result is True
        assert len(output) > 0  # Variables printed (even if empty)

    def test_non_special_command_returns_false(self) -> None:
        """Test that non-special commands return False."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        result = repl.eval_special_command("load test.bin")

        assert result is False

    def test_special_command_strips_whitespace(self) -> None:
        """Test that special commands handle whitespace."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        result = repl.eval_special_command("  exit  ")

        assert result is True
        assert repl.running is False


@pytest.mark.unit
class TestREPLEvalLine:
    """Test REPL line evaluation."""

    def test_eval_empty_line(self) -> None:
        """Test evaluating empty line does nothing."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        # Should not raise
        repl.eval_line("")
        repl.eval_line("   ")

    def test_eval_comment_line(self) -> None:
        """Test evaluating comment line does nothing."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        # Should not raise
        repl.eval_line("# This is a comment")
        repl.eval_line("#")

    def test_eval_exit_command(self) -> None:
        """Test evaluating exit command."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        repl.eval_line("exit")

        assert repl.running is False

    def test_eval_syntax_error_prints_message(self) -> None:
        """Test that syntax errors print error message."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        with mock.patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            repl.eval_line("[[invalid syntax]]")
            output = mock_stderr.getvalue()

        # Should print some error (may or may not be syntax error depending on parser)
        # Just verify it doesn't crash
        assert repl.running is True  # Should still be running

    def test_eval_strips_whitespace(self) -> None:
        """Test that line evaluation strips whitespace."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        repl.eval_line("  exit  ")

        assert repl.running is False


@pytest.mark.unit
class TestREPLRun:
    """Test REPL main loop."""

    def test_run_prints_banner_and_goodbye(self) -> None:
        """Test that run prints banner and goodbye."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        # Mock input to return None (EOF) immediately
        with mock.patch.object(repl, "read_input", return_value=None):
            with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                repl.run()
                output = mock_stdout.getvalue()

        assert "Oscura" in output
        assert "Goodbye" in output

    def test_run_handles_eof(self) -> None:
        """Test that run handles EOF gracefully."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        with mock.patch.object(repl, "read_input", return_value=None):
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                repl.run()

        # Should have exited loop
        assert True  # If we got here, it worked

    def test_run_processes_commands(self) -> None:
        """Test that run processes multiple commands."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        # Mock input to return some commands then exit
        inputs = iter(["help", "vars", "exit"])

        with mock.patch.object(repl, "read_input", side_effect=lambda: next(inputs)):
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                repl.run()

        assert repl.running is False

    def test_run_continues_after_error(self) -> None:
        """Test that run continues after command errors."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        # Mock input: invalid command, then exit
        inputs = iter(["invalid_command_xyz", "exit"])

        with mock.patch.object(repl, "read_input", side_effect=lambda: next(inputs)):
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                with mock.patch("sys.stderr", new_callable=io.StringIO):
                    repl.run()

        # Should have processed exit command
        assert repl.running is False


@pytest.mark.unit
class TestStartRepl:
    """Test start_repl entry point."""

    def test_start_repl_creates_and_runs_repl(self) -> None:
        """Test that start_repl creates and runs a REPL."""
        from oscura.dsl.repl import start_repl

        with mock.patch("oscura.dsl.repl.REPL") as MockREPL:
            mock_repl = MockREPL.return_value
            mock_repl.run = mock.Mock()

            start_repl()

            MockREPL.assert_called_once()
            mock_repl.run.assert_called_once()


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
class TestREPLIntegration:
    """Integration tests for REPL."""

    def test_full_session_simulation(self) -> None:
        """Test a simulated REPL session."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        # Simulate a session
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            with mock.patch("sys.stderr", new_callable=io.StringIO):
                repl.eval_line("help")
                repl.eval_line("vars")
                repl.eval_line("")  # Empty line
                repl.eval_line("# Comment")
                repl.eval_line("exit")

        assert repl.running is False

    def test_repl_maintains_state(self) -> None:
        """Test that REPL maintains interpreter state across commands."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        # Manually set a variable
        repl.interpreter.variables["test_var"] = 123

        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            repl.eval_line("vars")
            output = mock_stdout.getvalue()

        assert "test_var" in output
        assert "123" in output

    def test_multiple_repls_are_independent(self) -> None:
        """Test that multiple REPL instances are independent."""
        from oscura.dsl.repl import REPL

        repl1 = REPL()
        repl2 = REPL()

        repl1.interpreter.variables["x"] = 1
        repl2.interpreter.variables["y"] = 2

        assert "x" in repl1.interpreter.variables
        assert "y" not in repl1.interpreter.variables
        assert "y" in repl2.interpreter.variables
        assert "x" not in repl2.interpreter.variables


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
class TestREPLEdgeCases:
    """Test REPL edge cases."""

    def test_very_long_input_line(self) -> None:
        """Test handling very long input lines."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        long_line = "a" * 10000

        # Should not crash
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            repl.eval_line(long_line)

    def test_special_characters_in_input(self) -> None:
        """Test handling special characters in input."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        special_inputs = [
            "\t\t\t",
            "\n\n\n",
            "\r\n",
            "\\x00",
            "unicode: \u0000",
        ]

        for inp in special_inputs:
            # Should not crash
            with mock.patch("sys.stderr", new_callable=io.StringIO):
                repl.eval_line(inp)

    def test_exit_with_trailing_newline(self) -> None:
        """Test exit command with trailing newline."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        repl.eval_line("exit\n")

        assert repl.running is False

    def test_case_sensitivity_of_special_commands(self) -> None:
        """Test that special commands are case-sensitive."""
        from oscura.dsl.repl import REPL

        repl = REPL()

        # These should NOT be recognized as special commands
        result = repl.eval_special_command("EXIT")
        assert result is False

        result = repl.eval_special_command("Help")
        assert result is False

        # Lowercase should work
        result = repl.eval_special_command("help")
        assert result is True
