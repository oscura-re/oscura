"""Tests for Lua code builder."""

import pytest

from oscura.export.wireshark.lua_builder import LuaCodeBuilder

pytestmark = pytest.mark.unit


class TestLuaCodeBuilder:
    """Test Lua code builder functionality."""

    def test_basic_line_addition(self) -> None:
        """Test adding basic lines."""
        builder = LuaCodeBuilder()
        builder.add_line("local x = 1")
        builder.add_line("local y = 2")
        code = builder.to_string()
        assert "local x = 1" in code
        assert "local y = 2" in code

    def test_indentation(self) -> None:
        """Test indentation management."""
        builder = LuaCodeBuilder()
        builder.add_line("if true then")
        builder.indent()
        builder.add_line("print('indented')")
        builder.dedent()
        builder.add_line("end")
        code = builder.to_string()
        assert "    print('indented')" in code
        assert "if true then\n" in code

    def test_function_blocks(self) -> None:
        """Test function block generation."""
        builder = LuaCodeBuilder()
        builder.begin_function("test_func", ["arg1", "arg2"])
        builder.add_line("return arg1 + arg2")
        builder.end_function()
        code = builder.to_string()
        assert "function test_func(arg1, arg2)" in code
        assert "    return arg1 + arg2" in code
        assert "end" in code

    def test_if_blocks(self) -> None:
        """Test if statement generation."""
        builder = LuaCodeBuilder()
        builder.begin_if("x > 0")
        builder.add_line("print('positive')")
        builder.end_if()
        code = builder.to_string()
        assert "if x > 0 then" in code
        assert "    print('positive')" in code
        assert "end" in code

    def test_if_else_blocks(self) -> None:
        """Test if-else statement generation."""
        builder = LuaCodeBuilder()
        builder.begin_if("x > 0")
        builder.add_line("print('positive')")
        builder.add_else()
        builder.add_line("print('negative')")
        builder.end_if()
        code = builder.to_string()
        assert "if x > 0 then" in code
        assert "else" in code
        assert "print('positive')" in code
        assert "print('negative')" in code

    def test_if_elseif_blocks(self) -> None:
        """Test if-elseif statement generation."""
        builder = LuaCodeBuilder()
        builder.begin_if("x > 0")
        builder.add_line("print('positive')")
        builder.add_elseif("x < 0")
        builder.add_line("print('negative')")
        builder.end_if()
        code = builder.to_string()
        assert "if x > 0 then" in code
        assert "elseif x < 0 then" in code

    def test_for_loops(self) -> None:
        """Test for loop generation."""
        builder = LuaCodeBuilder()
        builder.begin_for("i", "1", "10")
        builder.add_line("print(i)")
        builder.end_for()
        code = builder.to_string()
        assert "for i = 1, 10 do" in code
        assert "    print(i)" in code
        assert "end" in code

    def test_variable_declaration(self) -> None:
        """Test variable declaration."""
        builder = LuaCodeBuilder()
        builder.add_variable("x", "10", local=True)
        builder.add_variable("y", "20", local=False)
        code = builder.to_string()
        assert "local x = 10" in code
        assert "y = 20" in code

    def test_comments(self) -> None:
        """Test comment generation."""
        builder = LuaCodeBuilder()
        builder.add_comment("This is a comment")
        code = builder.to_string()
        assert "-- This is a comment" in code

    def test_return_statement(self) -> None:
        """Test return statement."""
        builder = LuaCodeBuilder()
        builder.add_return("42")
        code = builder.to_string()
        assert "return 42" in code

    def test_blank_lines(self) -> None:
        """Test blank line insertion."""
        builder = LuaCodeBuilder()
        builder.add_line("line1")
        builder.add_blank_line()
        builder.add_line("line2")
        code = builder.to_string()
        lines = code.split("\n")
        assert len(lines) == 3
        assert lines[0] == "line1"
        assert lines[1] == ""
        assert lines[2] == "line2"

    def test_empty_line_no_indent(self) -> None:
        """Test that empty lines don't get indented."""
        builder = LuaCodeBuilder()
        builder.indent()
        builder.add_line("")
        code = builder.to_string()
        assert code == ""

    def test_custom_indent_size(self) -> None:
        """Test custom indentation size."""
        builder = LuaCodeBuilder(indent_size=2)
        builder.indent()
        builder.add_line("test")
        code = builder.to_string()
        assert "  test" in code

    def test_nested_indentation(self) -> None:
        """Test nested indentation."""
        builder = LuaCodeBuilder()
        builder.begin_function("outer", [])
        builder.begin_if("condition")
        builder.add_line("nested_code()")
        builder.end_if()
        builder.end_function()
        code = builder.to_string()
        assert "        nested_code()" in code  # 2 levels of indent

    def test_str_method(self) -> None:
        """Test __str__ method."""
        builder = LuaCodeBuilder()
        builder.add_line("test")
        assert str(builder) == builder.to_string()

    def test_dedent_at_zero_level(self) -> None:
        """Test dedent when already at zero level."""
        builder = LuaCodeBuilder()
        builder.dedent()  # Should not go negative
        builder.add_line("test")
        code = builder.to_string()
        assert code == "test"
