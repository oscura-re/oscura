"""Oscura DSL REPL (Read-Eval-Print Loop).

Interactive shell for Oscura DSL.
"""

import sys

from oscura.api.dsl.interpreter import Interpreter, InterpreterError
from oscura.api.dsl.parser import parse_dsl


class REPL:
    """Interactive DSL shell."""

    def __init__(self) -> None:
        """Initialize REPL with new interpreter."""
        self.interpreter = Interpreter()
        self.running = True

    def print_banner(self) -> None:
        """Print welcome banner."""
        print("Oscura DSL REPL v0.1.0")
        print("Type 'exit' or 'quit' to exit, 'help' for help")
        print()

    def print_help(self) -> None:
        """Print help message."""
        print("Oscura DSL Commands:")
        print("  load <filename>                  - Load a trace file")
        print("  filter <type> <params>           - Apply filter (lowpass, highpass, etc.)")
        print("  measure <name>                   - Measure property (rise_time, etc.)")
        print("  plot                             - Plot trace")
        print("  export <format>                  - Export data (json, csv, hdf5)")
        print("  glob(<pattern>)                  - Match files")
        print()
        print("Variables:")
        print("  $name = expression               - Assign variable")
        print("  $name                            - Reference variable")
        print()
        print("Pipelines:")
        print("  expr | command | command         - Chain operations")
        print()
        print("Loops:")
        print("  for $var in expr: statement      - Iterate")
        print()
        print("Special commands:")
        print("  help                             - Show this help")
        print("  vars                             - List variables")
        print("  exit, quit                       - Exit REPL")
        print()

    def print_variables(self) -> None:
        """Print current variables."""
        if not self.interpreter.variables:
            print("No variables defined")
            return

        print("Variables:")
        for name, value in self.interpreter.variables.items():
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 60:
                str_value = str_value[:57] + "..."
            print(f"  {name} = {str_value}")

    def read_input(self) -> str | None:
        """Read input line from user.

        Returns:
            Input line or None on EOF
        """
        try:
            return input("tk> ")
        except EOFError:
            return None

    def eval_special_command(self, line: str) -> bool:
        """Evaluate special REPL commands.

        Args:
            line: Input line

        Returns:
            True if special command was handled, False otherwise
        """
        line = line.strip()

        if line in ("exit", "quit"):
            self.running = False
            return True

        if line == "help":
            self.print_help()
            return True

        if line == "vars":
            self.print_variables()
            return True

        return False

    def eval_line(self, line: str) -> None:
        """Evaluate a single line of input.

        Args:
            line: Input line
        """
        line = line.strip()

        # Skip empty lines
        if not line:
            return

        # Skip comments
        if line.startswith("#"):
            return

        # Check for special commands
        if self.eval_special_command(line):
            return

        # Parse and execute
        try:
            ast = parse_dsl(line)
            self.interpreter.execute(ast)

            # Print result if expression statement (not assignment)
            if ast and hasattr(ast[-1], "expression"):
                # Assignment - don't print
                pass
            else:
                # Expression - could print result
                # For now, commands handle their own output
                pass

        except SyntaxError as e:
            print(f"Syntax error: {e}", file=sys.stderr)

        except InterpreterError as e:
            print(f"Error: {e}", file=sys.stderr)

        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)

    def run(self) -> None:
        """Run REPL loop."""
        self.print_banner()

        while self.running:
            line = self.read_input()

            if line is None:
                # EOF
                print()
                break

            self.eval_line(line)

        print("Goodbye!")


def start_repl() -> None:
    """Start interactive REPL.

    This is the main entry point for the DSL REPL.
    """
    repl = REPL()
    repl.run()


if __name__ == "__main__":
    start_repl()
