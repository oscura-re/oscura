"""Lua code builder for programmatic dissector generation.

This module provides a builder class for constructing Lua code with
proper indentation and structure.
"""


class LuaCodeBuilder:
    """Build Lua dissector code programmatically.

    This class helps construct well-formatted Lua code with proper
    indentation and structure management.

    Example:
        >>> builder = LuaCodeBuilder()
        >>> builder.add_comment("Initialize protocol")
        >>> builder.add_variable("proto", 'Proto("myproto", "My Protocol")')
        >>> builder.begin_function("dissector", ["buffer", "pinfo", "tree"])
        >>> builder.add_line("pinfo.cols.protocol = proto.name")
        >>> builder.end_function()
        >>> print(builder.to_string())
    """

    def __init__(self, indent_size: int = 4) -> None:
        """Initialize the builder.

        Args:
            indent_size: Number of spaces per indentation level
        """
        self.lines: list[str] = []
        self.indent_level = 0
        self.indent_size = indent_size

    def add_line(self, line: str) -> None:
        """Add a line with current indentation.

        Args:
            line: Line content (without indentation)
        """
        if line.strip():  # Don't indent empty lines
            indent = " " * (self.indent_level * self.indent_size)
            self.lines.append(f"{indent}{line}")
        else:
            self.lines.append("")

    def add_blank_line(self) -> None:
        """Add a blank line."""
        self.lines.append("")

    def add_comment(self, comment: str, prefix: str = "--") -> None:
        """Add a Lua comment.

        Args:
            comment: Comment text
            prefix: Comment prefix (default: "--")
        """
        self.add_line(f"{prefix} {comment}")

    def begin_function(self, name: str, params: list[str]) -> None:
        """Start a function definition.

        Args:
            name: Function name
            params: List of parameter names
        """
        param_list = ", ".join(params)
        self.add_line(f"function {name}({param_list})")
        self.indent_level += 1

    def end_function(self) -> None:
        """End a function definition."""
        self.indent_level -= 1
        self.add_line("end")

    def begin_if(self, condition: str) -> None:
        """Start an if statement.

        Args:
            condition: If condition
        """
        self.add_line(f"if {condition} then")
        self.indent_level += 1

    def add_else(self) -> None:
        """Add else clause."""
        self.indent_level -= 1
        self.add_line("else")
        self.indent_level += 1

    def add_elseif(self, condition: str) -> None:
        """Add elseif clause.

        Args:
            condition: Elseif condition
        """
        self.indent_level -= 1
        self.add_line(f"elseif {condition} then")
        self.indent_level += 1

    def end_if(self) -> None:
        """End an if statement."""
        self.indent_level -= 1
        self.add_line("end")

    def begin_for(self, loop_var: str, start: str, end: str) -> None:
        """Start a for loop.

        Args:
            loop_var: Loop variable name
            start: Start value
            end: End value
        """
        self.add_line(f"for {loop_var} = {start}, {end} do")
        self.indent_level += 1

    def end_for(self) -> None:
        """End a for loop."""
        self.indent_level -= 1
        self.add_line("end")

    def add_variable(self, name: str, value: str, local: bool = True) -> None:
        """Add a variable declaration.

        Args:
            name: Variable name
            value: Variable value
            local: Whether to use 'local' keyword
        """
        prefix = "local " if local else ""
        self.add_line(f"{prefix}{name} = {value}")

    def add_return(self, value: str) -> None:
        """Add a return statement.

        Args:
            value: Return value
        """
        self.add_line(f"return {value}")

    def indent(self) -> None:
        """Increase indentation level."""
        self.indent_level += 1

    def dedent(self) -> None:
        """Decrease indentation level."""
        if self.indent_level > 0:
            self.indent_level -= 1

    def to_string(self) -> str:
        """Generate final Lua code.

        Returns:
            Complete Lua code as a string
        """
        return "\n".join(self.lines)

    def __str__(self) -> str:
        """String representation."""
        return self.to_string()
