"""Oscura DSL Interpreter.

Executes parsed DSL programs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from oscura.api.dsl.parser import (
    Assignment,
    Command,
    Expression,
    ForLoop,
    FunctionCall,
    Literal,
    Pipeline,
    Statement,
    Variable,
    parse_dsl,
)


class InterpreterError(Exception):
    """DSL interpreter error.

    Raised when the interpreter encounters an error during execution,
    such as undefined variables, unknown commands, or invalid operations.
    """


class Interpreter:
    """DSL interpreter for Oscura commands.

    Executes parsed AST with Python implementations of DSL commands.
    """

    def __init__(self) -> None:
        """Initialize interpreter with empty environment."""
        self.variables: dict[str, Any] = {}
        self.commands: dict[str, Any] = {}
        self._register_builtin_commands()

    def _register_builtin_commands(self) -> None:
        """Register built-in DSL commands."""
        # Import oscura functions lazily to avoid circular imports
        self.commands["load"] = self._cmd_load
        self.commands["filter"] = self._cmd_filter
        self.commands["measure"] = self._cmd_measure
        self.commands["plot"] = self._cmd_plot
        self.commands["export"] = self._cmd_export
        self.commands["glob"] = self._cmd_glob

    def _cmd_load(self, *args: Any) -> Any:
        """Load command: load "filename"."""
        if len(args) != 1:
            raise InterpreterError("load requires exactly 1 argument: filename")

        filename = args[0]
        if not isinstance(filename, str):
            raise InterpreterError("load filename must be a string")

        # Lazy import to avoid circular dependency
        try:
            from oscura.loaders import load

            # Try to determine loader from extension
            path = Path(filename)
            if not path.exists():
                raise InterpreterError(f"File not found: {filename}")

            # Load the file using Oscura's loader
            return load(str(path))

        except ImportError as e:
            raise InterpreterError(f"oscura.loaders not available: {e}")
        except Exception as e:
            raise InterpreterError(f"Failed to load {filename}: {e}")

    def _cmd_filter(self, trace: Any, *args: Any) -> Any:
        """Filter command: filter lowpass 1000."""
        if len(args) < 1:
            raise InterpreterError("filter requires filter type argument")

        filter_type = args[0]
        if not isinstance(filter_type, str):
            raise InterpreterError("filter type must be a string")

        # Import filter functions
        try:
            from oscura.utils import filtering

            # Note: bandpass and bandstop have different signatures (low, high)
            # but we treat them as Any -> Any for DSL simplicity
            filter_map: dict[str, Callable[..., Any]] = {
                "lowpass": filtering.low_pass,
                "highpass": filtering.high_pass,
                "bandpass": filtering.band_pass,
                "bandstop": filtering.band_stop,
            }

            if filter_type not in filter_map:
                raise InterpreterError(
                    f"Unknown filter type: {filter_type}. Available: {', '.join(filter_map.keys())}"
                )

            # Get cutoff frequency (required for all filters)
            if len(args) < 2:
                raise InterpreterError(f"{filter_type} filter requires cutoff frequency")

            cutoff = args[1]
            if not isinstance(cutoff, (int, float)):
                raise InterpreterError("cutoff frequency must be a number")

            # Apply filter
            filter_func: Callable[..., Any] = filter_map[filter_type]
            return filter_func(trace, cutoff)

        except ImportError as e:
            raise InterpreterError(f"oscura.filtering not available: {e}")
        except Exception as e:
            raise InterpreterError(f"Filter failed: {e}")

    def _cmd_measure(self, trace: Any, *args: Any) -> Any:
        """Measure command: measure rise_time."""
        if len(args) < 1:
            raise InterpreterError("measure requires measurement name")

        measurement = args[0]
        if not isinstance(measurement, str):
            raise InterpreterError("measurement name must be a string")

        # Import measurement functions
        try:
            import oscura

            # Map measurement names to functions - access via getattr
            measure_map: dict[str, str] = {
                "frequency": "frequency",
                "period": "period",
                "duty_cycle": "duty_cycle",
                "rise_time": "rise_time",
                "fall_time": "fall_time",
                "amplitude": "amplitude",
                "rms": "rms",
                "mean": "mean",
                "peak_to_peak": "vpp",
            }

            if measurement not in measure_map:
                raise InterpreterError(
                    f"Unknown measurement: {measurement}. "
                    f"Available: {', '.join(measure_map.keys())}"
                )

            # Apply measurement - use getattr to access the function
            func_name = measure_map[measurement]
            measure_func = getattr(oscura, func_name)
            return measure_func(trace)

        except ImportError as e:
            raise InterpreterError(f"oscura measurements not available: {e}")
        except AttributeError as e:
            raise InterpreterError(f"Measurement function not found: {e}")
        except Exception as e:
            raise InterpreterError(f"Measurement failed: {e}")

    def _cmd_plot(self, trace: Any, *args: Any) -> Any:
        """Plot command: plot."""
        # Import visualization functions
        try:
            from oscura import visualization

            # Optional plot type (default: waveform)
            plot_type = "waveform"
            if len(args) > 0:
                plot_type = args[0]
                if not isinstance(plot_type, str):
                    raise InterpreterError("plot type must be a string")

            # Plot based on type - use getattr to access the plot function
            if plot_type == "waveform":
                plot_func = getattr(visualization, "plot", None)
                if plot_func is None:
                    raise InterpreterError("visualization.plot not available")
                plot_func(trace)
                return trace  # Return trace for chaining
            else:
                raise InterpreterError(f"Unknown plot type: {plot_type}. Available: waveform")

        except ImportError as e:
            raise InterpreterError(f"oscura.visualization not available: {e}")
        except Exception as e:
            raise InterpreterError(f"Plot failed: {e}")

    def _cmd_export(self, data: Any, *args: Any) -> Any:
        """Export command: export json "output.json"."""
        if len(args) < 1:
            raise InterpreterError("export requires format argument")

        format_type = args[0]
        if not isinstance(format_type, str):
            raise InterpreterError("export format must be a string")

        # Get optional filename
        filename = None
        if len(args) > 1:
            filename = args[1]
            if not isinstance(filename, str):
                raise InterpreterError("export filename must be a string")

        # Export functionality has been redesigned
        raise InterpreterError(
            "Data export has been redesigned. Use oscura.export.wireshark, "
            "oscura.export.kaitai_struct, or oscura.export.scapy_layer for protocol exports."
        )

    def _cmd_glob(self, pattern: str) -> list[str]:
        """Glob command: glob("*.csv")."""
        if not isinstance(pattern, str):
            raise InterpreterError("glob pattern must be a string")

        from glob import glob as glob_func

        return list(glob_func(pattern))

    def eval_expression(self, expr: Expression) -> Any:
        """Evaluate an expression.

        Args:
            expr: Expression AST node

        Returns:
            Evaluated result

        Raises:
            InterpreterError: On evaluation errors
        """
        # Literal value
        if isinstance(expr, Literal):
            return expr.value

        # Variable reference
        if isinstance(expr, Variable):
            if expr.name not in self.variables:
                raise InterpreterError(f"Undefined variable: {expr.name} at line {expr.line}")
            return self.variables[expr.name]

        # Function call
        if isinstance(expr, FunctionCall):
            return self.eval_function_call(expr)

        # Command
        if isinstance(expr, Command):
            return self.eval_command(expr, None)

        # Pipeline
        if isinstance(expr, Pipeline):
            return self.eval_pipeline(expr)

        raise InterpreterError(f"Unknown expression type: {type(expr).__name__}")

    def eval_function_call(self, func: FunctionCall) -> Any:
        """Evaluate function call."""
        if func.name not in self.commands:
            raise InterpreterError(f"Unknown function: {func.name} at line {func.line}")

        # Evaluate arguments
        args = [self.eval_expression(arg) for arg in func.args]

        # Call command function
        return self.commands[func.name](*args)

    def eval_command(self, cmd: Command, input_data: Any | None) -> Any:
        """Evaluate command with optional piped input.

        Args:
            cmd: Command AST node
            input_data: Input from previous pipeline stage (or None)

        Returns:
            Command result

        Raises:
            InterpreterError: If command is unknown
        """
        if cmd.name not in self.commands:
            raise InterpreterError(f"Unknown command: {cmd.name} at line {cmd.line}")

        # Evaluate arguments
        args = [self.eval_expression(arg) for arg in cmd.args]

        # If there's input data, prepend it as first argument
        if input_data is not None:
            args = [input_data, *args]

        # Call command function
        return self.commands[cmd.name](*args)

    def eval_pipeline(self, pipeline: Pipeline) -> Any:
        """Evaluate pipeline of commands.

        Args:
            pipeline: Pipeline AST node

        Returns:
            Final pipeline result

        Raises:
            InterpreterError: If pipeline stage is invalid
        """
        result = None

        for i, stage in enumerate(pipeline.stages):
            if i == 0:
                # First stage - no input
                if isinstance(stage, Command):
                    result = self.eval_command(stage, None)
                else:
                    result = self.eval_expression(stage)
            # Subsequent stages - pipe input from previous
            elif isinstance(stage, Command):
                result = self.eval_command(stage, result)
            else:
                raise InterpreterError(
                    f"Pipeline stage {i + 1} must be a command, got {type(stage).__name__}"
                )

        return result

    def eval_statement(self, stmt: Statement) -> Any:
        """Execute a statement.

        Args:
            stmt: Statement AST node

        Returns:
            Result of statement execution (None for Assignment and ForLoop, value for expressions)

        Raises:
            InterpreterError: On execution errors
        """
        # Assignment
        if isinstance(stmt, Assignment):
            value = self.eval_expression(stmt.expression)
            self.variables[stmt.variable] = value
            return None

        # For loop
        if isinstance(stmt, ForLoop):
            self.eval_for_loop(stmt)
            return None

        # Expression statement (pipeline)
        if isinstance(stmt, Pipeline):
            return self.eval_pipeline(stmt)

        # Function call as statement (e.g., in for loop body: process($f))
        if isinstance(stmt, FunctionCall):
            return self.eval_function_call(stmt)

        # All Statement types covered above (Assignment, ForLoop, Pipeline, FunctionCall)
        # This line is unreachable if type system is correct, but kept for runtime safety
        raise InterpreterError(f"Unknown statement type: {type(stmt).__name__}")

    def eval_for_loop(self, loop: ForLoop) -> None:
        """Execute for loop.

        Args:
            loop: ForLoop AST node

        Raises:
            InterpreterError: If iterable is not iterable
        """
        # Evaluate iterable
        iterable = self.eval_expression(loop.iterable)

        if not hasattr(iterable, "__iter__"):
            raise InterpreterError(
                f"For loop iterable is not iterable: {type(iterable).__name__} at line {loop.line}"
            )

        # Execute body for each item
        for item in iterable:
            # Set loop variable
            self.variables[loop.variable] = item

            # Execute body statements
            for stmt in loop.body:
                self.eval_statement(stmt)

    def execute(self, statements: list[Statement]) -> None:
        """Execute a program (list of statements).

        Args:
            statements: AST (list of statements)
        """
        for stmt in statements:
            self.eval_statement(stmt)

    def execute_source(self, source: str) -> None:
        """Parse and execute DSL source code.

        Args:
            source: DSL source code
        """
        ast = parse_dsl(source)
        self.execute(ast)


def execute_dsl(source: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute Oscura DSL source code.

    Args:
        source: DSL source code
        variables: Optional initial variables

    Returns:
        Final variable environment after execution
    """
    interpreter = Interpreter()

    # Set initial variables
    if variables:
        interpreter.variables.update(variables)

    # Execute
    interpreter.execute_source(source)

    return interpreter.variables
