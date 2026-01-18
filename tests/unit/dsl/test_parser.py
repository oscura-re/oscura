"""Unit tests for the Oscura DSL parser.

Tests the lexer and parser components of the domain-specific
language for trace analysis workflows.
"""

import pytest

from oscura.dsl.parser import (
    Assignment,
    Command,
    ForLoop,
    FunctionCall,
    Lexer,
    Literal,
    Pipeline,
    Token,
    TokenType,
    Variable,
    parse_dsl,
)

pytestmark = pytest.mark.unit


class TestToken:
    """Tests for the Token dataclass."""

    def test_token_creation(self):
        """Test creating a Token."""
        token = Token(
            type=TokenType.STRING,
            value="hello",
            line=1,
            column=5,
        )

        assert token.type == TokenType.STRING
        assert token.value == "hello"
        assert token.line == 1
        assert token.column == 5


class TestTokenType:
    """Tests for the TokenType enum."""

    def test_all_token_types_exist(self):
        """Test that all expected token types exist."""
        # Literals
        assert TokenType.STRING
        assert TokenType.NUMBER
        assert TokenType.VARIABLE
        assert TokenType.IDENTIFIER

        # Operators
        assert TokenType.PIPE
        assert TokenType.ASSIGN
        assert TokenType.COMMA

        # Keywords
        assert TokenType.LOAD
        assert TokenType.FILTER
        assert TokenType.MEASURE
        assert TokenType.PLOT
        assert TokenType.EXPORT
        assert TokenType.FOR
        assert TokenType.IN
        assert TokenType.GLOB

        # Structural
        assert TokenType.LPAREN
        assert TokenType.RPAREN
        assert TokenType.COLON
        assert TokenType.NEWLINE
        assert TokenType.INDENT
        assert TokenType.DEDENT
        assert TokenType.EOF


class TestLexer:
    """Tests for the Lexer class."""

    def test_lexer_creation(self):
        """Test creating a Lexer."""
        lexer = Lexer("test input")
        assert lexer.text == "test input"
        assert lexer.pos == 0
        assert lexer.line == 1
        assert lexer.column == 1

    def test_tokenize_simple_string(self):
        """Test tokenizing a simple string."""
        lexer = Lexer('"hello"')
        tokens = lexer.tokenize()

        assert len(tokens) >= 2  # STRING + EOF
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"

    def test_tokenize_single_quoted_string(self):
        """Test tokenizing single-quoted string."""
        lexer = Lexer("'hello'")
        tokens = lexer.tokenize()

        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"

    def test_tokenize_string_with_escapes(self):
        """Test tokenizing string with escape sequences."""
        lexer = Lexer('"hello\\nworld"')
        tokens = lexer.tokenize()

        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello\nworld"

    def test_tokenize_integer(self):
        """Test tokenizing an integer."""
        lexer = Lexer("42")
        tokens = lexer.tokenize()

        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 42
        assert isinstance(tokens[0].value, int)

    def test_tokenize_float(self):
        """Test tokenizing a float."""
        lexer = Lexer("3.14")
        tokens = lexer.tokenize()

        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 3.14
        assert isinstance(tokens[0].value, float)

    def test_tokenize_scientific_notation(self):
        """Test tokenizing scientific notation."""
        lexer = Lexer("1e6")
        tokens = lexer.tokenize()

        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 1e6

    def test_tokenize_variable(self):
        """Test tokenizing a variable."""
        lexer = Lexer("$myvar")
        tokens = lexer.tokenize()

        assert tokens[0].type == TokenType.VARIABLE
        assert tokens[0].value == "$myvar"

    def test_tokenize_identifier(self):
        """Test tokenizing an identifier."""
        lexer = Lexer("myfunction")
        tokens = lexer.tokenize()

        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "myfunction"

    def test_tokenize_keywords(self):
        """Test tokenizing keywords."""
        keywords = ["load", "filter", "measure", "plot", "export", "for", "in", "glob"]

        for keyword in keywords:
            lexer = Lexer(keyword)
            tokens = lexer.tokenize()
            assert tokens[0].type != TokenType.IDENTIFIER

    def test_tokenize_operators(self):
        """Test tokenizing operators."""
        lexer = Lexer("| = , : ( )")
        tokens = lexer.tokenize()

        types = [t.type for t in tokens[:-1]]  # Exclude EOF
        assert TokenType.PIPE in types
        assert TokenType.ASSIGN in types
        assert TokenType.COMMA in types
        assert TokenType.COLON in types
        assert TokenType.LPAREN in types
        assert TokenType.RPAREN in types

    def test_tokenize_newline(self):
        """Test tokenizing newlines."""
        lexer = Lexer("a\nb")
        tokens = lexer.tokenize()

        types = [t.type for t in tokens]
        assert TokenType.NEWLINE in types

    def test_tokenize_skips_whitespace(self):
        """Test that lexer skips whitespace."""
        lexer = Lexer("   hello   ")
        tokens = lexer.tokenize()

        # Leading whitespace at start of line is treated as indentation
        # so we get INDENT, IDENTIFIER, DEDENT, EOF
        assert tokens[0].type == TokenType.INDENT
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[1].value == "hello"

    def test_tokenize_skips_comments(self):
        """Test that lexer skips comments."""
        lexer = Lexer("hello # this is a comment")
        tokens = lexer.tokenize()

        # Should only have 'hello' and EOF
        assert len(tokens) == 2
        assert tokens[0].value == "hello"

    def test_tokenize_indentation(self):
        """Test tokenizing indentation."""
        source = "a:\n    b"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        types = [t.type for t in tokens]
        assert TokenType.INDENT in types

    def test_tokenize_dedentation(self):
        """Test tokenizing dedentation."""
        source = "a:\n    b\nc"
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        types = [t.type for t in tokens]
        assert TokenType.DEDENT in types

    def test_tokenize_eof(self):
        """Test that tokenize ends with EOF."""
        lexer = Lexer("")
        tokens = lexer.tokenize()

        assert tokens[-1].type == TokenType.EOF

    def test_tokenize_unterminated_string_error(self):
        """Test error on unterminated string."""
        lexer = Lexer('"hello')

        with pytest.raises(SyntaxError, match="Unterminated string"):
            lexer.tokenize()

    def test_tokenize_unexpected_character_error(self):
        """Test error on unexpected character."""
        lexer = Lexer("@")

        with pytest.raises(SyntaxError, match="Unexpected character"):
            lexer.tokenize()

    def test_tokenize_tabs_as_spaces(self):
        """Test that tabs are treated as 4 spaces."""
        source = "a:\n\tb"  # Tab indentation
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        types = [t.type for t in tokens]
        assert TokenType.INDENT in types


class TestParser:
    """Tests for the Parser class."""

    def test_parse_simple_command(self):
        """Test parsing a simple command."""
        ast = parse_dsl('load "file.wfm"')

        assert len(ast) == 1
        assert isinstance(ast[0], Command)
        assert ast[0].name == "load"
        assert len(ast[0].args) == 1

    def test_parse_variable_assignment(self):
        """Test parsing variable assignment."""
        ast = parse_dsl('$data = load "file.wfm"')

        assert len(ast) == 1
        assert isinstance(ast[0], Assignment)
        assert ast[0].variable == "$data"
        assert isinstance(ast[0].expression, Command)

    def test_parse_pipeline(self):
        """Test parsing a pipeline."""
        ast = parse_dsl("$data | filter | measure")

        assert len(ast) == 1
        assert isinstance(ast[0], Pipeline)
        assert len(ast[0].stages) == 3

    def test_parse_function_call(self):
        """Test parsing a function call with parentheses."""
        ast = parse_dsl('glob("*.wfm")')

        assert len(ast) == 1
        assert isinstance(ast[0], FunctionCall)
        assert ast[0].name == "glob"
        assert len(ast[0].args) == 1

    def test_parse_literal_string(self):
        """Test parsing string literal."""
        ast = parse_dsl('"hello"')

        assert len(ast) == 1
        assert isinstance(ast[0], Literal)
        assert ast[0].value == "hello"

    def test_parse_literal_number(self):
        """Test parsing number literal."""
        ast = parse_dsl("42")

        assert len(ast) == 1
        assert isinstance(ast[0], Literal)
        assert ast[0].value == 42

    def test_parse_variable_reference(self):
        """Test parsing variable reference."""
        ast = parse_dsl("$myvar")

        assert len(ast) == 1
        assert isinstance(ast[0], Variable)
        assert ast[0].name == "$myvar"

    def test_parse_for_loop_single_line(self):
        """Test parsing single-line for loop."""
        ast = parse_dsl('for $f in glob("*.wfm"): load $f')

        assert len(ast) == 1
        assert isinstance(ast[0], ForLoop)
        assert ast[0].variable == "$f"
        assert len(ast[0].body) == 1

    def test_parse_for_loop_multi_line(self):
        """Test parsing multi-line for loop."""
        source = """for $f in glob("*.wfm"):
    load $f
    measure $f"""
        ast = parse_dsl(source)

        assert len(ast) == 1
        assert isinstance(ast[0], ForLoop)
        assert len(ast[0].body) == 2

    def test_parse_multiple_statements(self):
        """Test parsing multiple statements."""
        source = """$a = load "a.wfm"
$b = load "b.wfm"
measure $a"""
        ast = parse_dsl(source)

        assert len(ast) == 3
        assert isinstance(ast[0], Assignment)
        assert isinstance(ast[1], Assignment)
        assert isinstance(ast[2], Command)

    def test_parse_command_with_multiple_args(self):
        """Test parsing command with multiple arguments."""
        ast = parse_dsl('filter "lowpass" 1000')

        assert len(ast) == 1
        assert isinstance(ast[0], Command)
        assert len(ast[0].args) == 2

    def test_parse_function_with_multiple_args(self):
        """Test parsing function with multiple arguments."""
        ast = parse_dsl("func(1, 2, 3)")

        assert len(ast) == 1
        assert isinstance(ast[0], FunctionCall)
        assert len(ast[0].args) == 3

    def test_parse_empty_input(self):
        """Test parsing empty input."""
        ast = parse_dsl("")
        assert ast == []

    def test_parse_blank_lines(self):
        """Test parsing with blank lines."""
        source = """load "a.wfm"

load "b.wfm"
"""
        ast = parse_dsl(source)
        assert len(ast) == 2

    def test_parse_comments_only(self):
        """Test parsing comments-only input."""
        source = """# This is a comment
# Another comment"""
        ast = parse_dsl(source)
        assert ast == []

    def test_parse_mixed_comments(self):
        """Test parsing mixed code and comments."""
        source = """# Load data
load "file.wfm"  # with comment"""
        ast = parse_dsl(source)
        assert len(ast) == 1

    def test_parse_error_unexpected_token(self):
        """Test parse error on unexpected token."""
        with pytest.raises(SyntaxError):
            parse_dsl("( )")

    def test_parse_keywords_as_commands(self):
        """Test that keywords work as commands."""
        # All DSL keywords should work as commands
        for keyword in ["load", "filter", "measure", "plot", "export"]:
            ast = parse_dsl(f'{keyword} "arg"')
            assert len(ast) == 1
            assert isinstance(ast[0], Command)
            assert ast[0].name == keyword


class TestASTNodes:
    """Tests for AST node dataclasses."""

    def test_assignment_node(self):
        """Test Assignment node creation."""
        var = Variable(name="$x", line=1, column=1)
        expr = Literal(value=42, line=1, column=6)

        node = Assignment(variable="$x", expression=expr, line=1, column=1)

        assert node.variable == "$x"
        assert node.expression is expr

    def test_pipeline_node(self):
        """Test Pipeline node creation."""
        stages = [
            Variable(name="$data", line=1, column=1),
            Command(name="filter", args=[], line=1, column=10),
        ]

        node = Pipeline(stages=stages, line=1, column=1)

        assert len(node.stages) == 2

    def test_command_node(self):
        """Test Command node creation."""
        args = [Literal(value="file.wfm", line=1, column=6)]

        node = Command(name="load", args=args, line=1, column=1)

        assert node.name == "load"
        assert len(node.args) == 1

    def test_function_call_node(self):
        """Test FunctionCall node creation."""
        args = [Literal(value="*.wfm", line=1, column=6)]

        node = FunctionCall(name="glob", args=args, line=1, column=1)

        assert node.name == "glob"
        assert len(node.args) == 1

    def test_variable_node(self):
        """Test Variable node creation."""
        node = Variable(name="$myvar", line=1, column=1)

        assert node.name == "$myvar"

    def test_literal_node(self):
        """Test Literal node creation."""
        string_node = Literal(value="hello", line=1, column=1)
        number_node = Literal(value=42, line=1, column=1)
        float_node = Literal(value=3.14, line=1, column=1)

        assert string_node.value == "hello"
        assert number_node.value == 42
        assert float_node.value == 3.14

    def test_for_loop_node(self):
        """Test ForLoop node creation."""
        iterable = FunctionCall(name="glob", args=[], line=1, column=10)
        body = [Command(name="load", args=[], line=2, column=5)]

        node = ForLoop(variable="$f", iterable=iterable, body=body, line=1, column=1)

        assert node.variable == "$f"
        assert node.iterable is iterable
        assert len(node.body) == 1


class TestParseDslFunction:
    """Tests for the parse_dsl convenience function."""

    def test_parse_dsl_simple(self):
        """Test parse_dsl with simple input."""
        ast = parse_dsl('load "test.wfm"')
        assert len(ast) == 1

    def test_parse_dsl_complex(self):
        """Test parse_dsl with complex input."""
        source = """$data = load "capture.wfm"
$filtered = $data | filter "lowpass" 1e6
for $m in glob("measurements"):
    measure $filtered
export $data "output.csv"
"""
        ast = parse_dsl(source)
        assert len(ast) >= 3

    def test_parse_dsl_syntax_error(self):
        """Test parse_dsl raises SyntaxError on bad input."""
        with pytest.raises(SyntaxError):
            parse_dsl("@invalid!")


class TestLexerEdgeCases:
    """Edge case tests for the Lexer."""

    def test_lexer_empty_string(self):
        """Test lexing empty string."""
        lexer = Lexer("")
        tokens = lexer.tokenize()

        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_lexer_only_whitespace(self):
        """Test lexing only whitespace."""
        lexer = Lexer("   \t   ")
        tokens = lexer.tokenize()

        assert tokens[-1].type == TokenType.EOF

    def test_lexer_multiple_dedents(self):
        """Test lexing multiple dedent levels."""
        source = """a:
    b:
        c
d"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        dedent_count = sum(1 for t in tokens if t.type == TokenType.DEDENT)
        assert dedent_count >= 1

    def test_lexer_identifier_with_underscore(self):
        """Test lexing identifier with underscore."""
        lexer = Lexer("my_function_name")
        tokens = lexer.tokenize()

        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "my_function_name"

    def test_lexer_identifier_with_numbers(self):
        """Test lexing identifier with numbers."""
        lexer = Lexer("func123")
        tokens = lexer.tokenize()

        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "func123"


class TestParserEdgeCases:
    """Edge case tests for the Parser."""

    def test_parser_deeply_nested_pipeline(self):
        """Test parsing deeply nested pipeline."""
        ast = parse_dsl("$a | $b | $c | $d | $e")

        assert len(ast) == 1
        assert isinstance(ast[0], Pipeline)
        assert len(ast[0].stages) == 5

    def test_parser_nested_for_loops(self):
        """Test parsing nested for loops."""
        source = """for $a in items:
    for $b in subitems:
        process $a $b"""
        ast = parse_dsl(source)

        assert len(ast) == 1
        assert isinstance(ast[0], ForLoop)
        assert isinstance(ast[0].body[0], ForLoop)

    def test_parser_line_column_tracking(self):
        """Test that parser tracks line and column numbers."""
        source = """line1
line2"""
        ast = parse_dsl(source)

        assert ast[0].line == 1
        assert ast[1].line == 2
