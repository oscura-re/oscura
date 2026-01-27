"""Oscura DSL Parser.

Implements simple domain-specific language for trace analysis workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Union


class TokenType(Enum):
    """Token types for DSL lexer."""

    # Literals
    STRING = auto()
    NUMBER = auto()
    VARIABLE = auto()
    IDENTIFIER = auto()

    # Operators
    PIPE = auto()
    ASSIGN = auto()
    COMMA = auto()

    # Keywords
    LOAD = auto()
    FILTER = auto()
    MEASURE = auto()
    PLOT = auto()
    EXPORT = auto()
    FOR = auto()
    IN = auto()
    GLOB = auto()

    # Structural
    LPAREN = auto()
    RPAREN = auto()
    COLON = auto()
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()


@dataclass
class Token:
    """Lexical token."""

    type: TokenType
    value: Any
    line: int
    column: int


class Lexer:
    """Tokenizer for Oscura DSL.

    Breaks input text into tokens for parsing.
    Supports indentation-based block structure (Python-style).
    """

    KEYWORDS = {
        "load": TokenType.LOAD,
        "filter": TokenType.FILTER,
        "measure": TokenType.MEASURE,
        "plot": TokenType.PLOT,
        "export": TokenType.EXPORT,
        "for": TokenType.FOR,
        "in": TokenType.IN,
        "glob": TokenType.GLOB,
    }

    def __init__(self, text: str):
        """Initialize lexer with input text.

        Args:
            text: DSL source code
        """
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: list[Token] = []
        # Indentation tracking
        self.indent_stack: list[int] = [0]
        self.at_line_start = True

    def current_char(self) -> str | None:
        """Get current character without advancing."""
        if self.pos >= len(self.text):
            return None
        return self.text[self.pos]

    def peek_char(self, offset: int = 1) -> str | None:
        """Peek ahead at character."""
        pos = self.pos + offset
        if pos >= len(self.text):
            return None
        return self.text[pos]

    def advance(self) -> None:
        """Advance position and update line/column."""
        if self.pos < len(self.text) and self.text[self.pos] == "\n":
            self.line += 1
            self.column = 1
            self.at_line_start = True
        else:
            self.column += 1
        self.pos += 1

    def skip_whitespace(self) -> None:
        """Skip whitespace except newlines."""
        while self.current_char() and self.current_char() in " \t\r":  # type: ignore[operator]
            self.advance()

    def skip_comment(self) -> None:
        """Skip # comment to end of line."""
        if self.current_char() == "#":
            while self.current_char() and self.current_char() != "\n":
                self.advance()

    def measure_indent(self) -> int:
        """Measure indentation at current position (after newline).

        Returns:
            Number of spaces of indentation (tabs count as 4 spaces)
        """
        indent = 0
        start_pos = self.pos

        while self.current_char() and self.current_char() in " \t":  # type: ignore[operator]
            if self.current_char() == " ":
                indent += 1
            elif self.current_char() == "\t":
                indent += 4  # Tab = 4 spaces
            self.pos += 1
            self.column += 1

        # Check if rest of line is blank or comment
        if self.current_char() == "#" or self.current_char() == "\n" or self.current_char() is None:
            # Blank line or comment-only line - reset position and return -1
            self.pos = start_pos
            self.column = 1
            return -1  # Signal to ignore this line for indentation

        return indent

    def read_string(self) -> str:
        """Read quoted string literal."""
        quote_char = self.current_char()
        self.advance()  # Skip opening quote

        chars = []
        while self.current_char() and self.current_char() != quote_char:
            if self.current_char() == "\\":
                self.advance()
                # Simple escape sequences
                escape_map = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'"}
                if self.current_char() in escape_map:
                    chars.append(escape_map[self.current_char()])  # type: ignore[index]
                else:
                    chars.append(self.current_char() or "")
            else:
                chars.append(self.current_char() or "")
            self.advance()

        if not self.current_char():
            raise SyntaxError(f"Unterminated string at line {self.line}")

        self.advance()  # Skip closing quote
        return "".join(chars)

    def read_number(self) -> int | float:
        """Read numeric literal."""
        chars = []
        has_dot = False
        has_exp = False

        while self.current_char() and (
            self.current_char().isdigit() or self.current_char() in ".eE+-"  # type: ignore[union-attr, operator]
        ):
            if self.current_char() == ".":
                if has_dot:
                    break
                has_dot = True
            elif self.current_char() in "eE":  # type: ignore[operator]
                if has_exp:
                    break
                has_exp = True
            chars.append(self.current_char())
            self.advance()

        num_str = "".join(chars)  # type: ignore[arg-type]
        return float(num_str) if has_dot or has_exp else int(num_str)

    def read_identifier(self) -> str:
        """Read identifier or keyword."""
        chars = []
        current = self.current_char()
        while current and (current.isalnum() or current in "_"):
            chars.append(current)
            self.advance()
            current = self.current_char()
        return "".join(chars)

    def read_variable(self) -> str:
        """Read variable name ($varname)."""
        self.advance()  # Skip $
        return "$" + self.read_identifier()

    def emit_indent_tokens(self, indent: int) -> None:
        """Emit INDENT/DEDENT tokens based on indentation change.

        Args:
            indent: Current line's indentation level

        Raises:
            SyntaxError: If indentation is inconsistent.
        """
        current_indent = self.indent_stack[-1]

        if indent > current_indent:
            # Increased indentation
            self.indent_stack.append(indent)
            self.tokens.append(Token(TokenType.INDENT, indent, self.line, 1))
        elif indent < current_indent:
            # Decreased indentation - may need multiple DEDENTs
            while self.indent_stack and indent < self.indent_stack[-1]:
                self.indent_stack.pop()
                self.tokens.append(Token(TokenType.DEDENT, indent, self.line, 1))

            # Check for inconsistent indentation
            if self.indent_stack and indent != self.indent_stack[-1]:
                raise SyntaxError(
                    f"Inconsistent indentation at line {self.line}: "
                    f"got {indent} spaces, expected {self.indent_stack[-1]}"
                )

    def tokenize(self) -> list[Token]:
        """Tokenize entire input.

        Returns:
            List of tokens

        Raises:
            SyntaxError: On lexical errors
        """
        while self.pos < len(self.text):
            if not self._process_line_start():
                break

            self.skip_whitespace()
            self.skip_comment()

            if not self.current_char():
                break

            self._process_token()

        self._finalize_tokens()
        return self.tokens

    def _process_line_start(self) -> bool:
        """Process indentation at line start.

        Returns:
            False if EOF reached, True otherwise.
        """
        if self.at_line_start:
            self.at_line_start = False
            indent = self.measure_indent()

            if indent == -1:
                return self._skip_blank_line()
            else:
                self.emit_indent_tokens(indent)
        return True

    def _skip_blank_line(self) -> bool:
        """Skip blank or comment-only line.

        Returns:
            False if EOF, True otherwise.
        """
        self.skip_whitespace()
        self.skip_comment()
        if self.current_char() == "\n":
            self.advance()
            return True
        return self.current_char() is not None

    def _process_token(self) -> None:
        """Process and emit next token."""
        line, col = self.line, self.column
        char = self.current_char()

        if char == "\n":
            self._emit_newline(line, col)
        elif char in "\"'":  # type: ignore[operator]
            self._emit_string(line, col)
        elif self._is_number_start(char):
            self._emit_number(line, col)
        elif char == "$":
            self._emit_variable(line, col)
        elif self._is_single_char_token(char):
            self._emit_single_char(char, line, col)
        elif char.isalpha() or char == "_":  # type: ignore[union-attr]
            self._emit_identifier(line, col)
        else:
            raise SyntaxError(f"Unexpected character '{char}' at line {line}, column {col}")

    def _is_number_start(self, char: str | None) -> bool:
        """Check if character starts a number."""
        if not char:
            return False
        next_char = self.peek_char()
        return char.isdigit() or (char == "." and next_char is not None and next_char.isdigit())

    def _is_single_char_token(self, char: str | None) -> bool:
        """Check if character is a single-character token."""
        if char is None:
            return False
        return char in "|=,:()"

    def _emit_newline(self, line: int, col: int) -> None:
        """Emit newline token."""
        self.tokens.append(Token(TokenType.NEWLINE, "\n", line, col))
        self.advance()

    def _emit_string(self, line: int, col: int) -> None:
        """Emit string token."""
        value = self.read_string()
        self.tokens.append(Token(TokenType.STRING, value, line, col))

    def _emit_number(self, line: int, col: int) -> None:
        """Emit number token."""
        value = self.read_number()
        self.tokens.append(Token(TokenType.NUMBER, value, line, col))

    def _emit_variable(self, line: int, col: int) -> None:
        """Emit variable token."""
        value = self.read_variable()
        self.tokens.append(Token(TokenType.VARIABLE, value, line, col))

    def _emit_single_char(self, char: str | None, line: int, col: int) -> None:
        """Emit single-character token."""
        token_map = {
            "|": TokenType.PIPE,
            "=": TokenType.ASSIGN,
            ",": TokenType.COMMA,
            ":": TokenType.COLON,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
        }
        self.tokens.append(Token(token_map[char], char, line, col))  # type: ignore[index]
        self.advance()

    def _emit_identifier(self, line: int, col: int) -> None:
        """Emit identifier or keyword token."""
        ident = self.read_identifier()
        token_type = self.KEYWORDS.get(ident.lower(), TokenType.IDENTIFIER)
        self.tokens.append(Token(token_type, ident, line, col))

    def _finalize_tokens(self) -> None:
        """Emit remaining DEDENT and EOF tokens."""
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.tokens.append(Token(TokenType.DEDENT, 0, self.line, self.column))
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))


@dataclass
class ASTNode:
    """Base class for AST nodes."""

    line: int
    column: int


@dataclass
class Assignment(ASTNode):
    """Variable assignment: $var = expr."""

    variable: str
    expression: Expression


@dataclass
class Pipeline(ASTNode):
    """Pipeline expression: expr | command | command."""

    stages: list[Expression]


@dataclass
class Command(ASTNode):
    """Command invocation: command arg1 arg2."""

    name: str
    args: list[Expression]


@dataclass
class FunctionCall(ASTNode):
    """Function call: func(arg1, arg2)."""

    name: str
    args: list[Expression]


@dataclass
class Variable(ASTNode):
    """Variable reference: $var."""

    name: str


@dataclass
class Literal(ASTNode):
    """Literal value: string, number."""

    value: str | int | float


@dataclass
class ForLoop(ASTNode):
    """For loop: for $var in expr: body."""

    variable: str
    iterable: Expression
    body: list[Statement]


# Type aliases
Expression = Union[Pipeline, Command, FunctionCall, Variable, Literal]
Statement = Union[Assignment, Pipeline, ForLoop, FunctionCall]


class Parser:
    """Recursive descent parser for Oscura DSL.

    Parses token stream into abstract syntax tree.
    Supports indentation-based block structure.
    """

    def __init__(self, tokens: list[Token]):
        """Initialize parser with token list.

        Args:
            tokens: Token list from lexer
        """
        self.tokens = tokens
        self.pos = 0

    def current_token(self) -> Token:
        """Get current token."""
        if self.pos >= len(self.tokens):
            return self.tokens[-1]  # EOF
        return self.tokens[self.pos]

    def peek_token(self, offset: int = 1) -> Token:
        """Peek ahead at token."""
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]  # EOF
        return self.tokens[pos]

    def advance(self) -> None:
        """Advance to next token."""
        if self.pos < len(self.tokens):
            self.pos += 1

    def expect(self, token_type: TokenType) -> Token:
        """Expect specific token type and advance.

        Args:
            token_type: Expected token type

        Returns:
            The token

        Raises:
            SyntaxError: If token type doesn't match
        """
        token = self.current_token()
        if token.type != token_type:
            raise SyntaxError(
                f"Expected {token_type.name}, got {token.type.name} "
                f"at line {token.line}, column {token.column}"
            )
        self.advance()
        return token

    def skip_newlines(self) -> None:
        """Skip optional newlines."""
        while self.current_token().type == TokenType.NEWLINE:
            self.advance()

    def parse(self) -> list[Statement]:
        """Parse complete program.

        Returns:
            List of statements (AST)

        Note:
            May raise SyntaxError on parse errors via parse_statement().
        """
        statements = []

        while self.current_token().type != TokenType.EOF:
            self.skip_newlines()
            if self.current_token().type == TokenType.EOF:
                break

            stmt = self.parse_statement()
            statements.append(stmt)
            self.skip_newlines()

        return statements

    def parse_statement(self) -> Statement:
        """Parse a single statement."""
        # For loop
        if self.current_token().type == TokenType.FOR:
            return self.parse_for_loop()

        # Assignment or expression
        if self.current_token().type == TokenType.VARIABLE:
            if self.peek_token().type == TokenType.ASSIGN:
                return self.parse_assignment()

        # Pipeline expression
        return self.parse_pipeline()  # type: ignore[return-value]

    def parse_assignment(self) -> Assignment:
        """Parse variable assignment."""
        token = self.current_token()
        var_token = self.expect(TokenType.VARIABLE)
        self.expect(TokenType.ASSIGN)
        expr = self.parse_pipeline()

        return Assignment(
            variable=var_token.value,
            expression=expr,
            line=token.line,
            column=token.column,
        )

    def parse_pipeline(self) -> Expression:
        """Parse pipeline expression."""
        stages = [self.parse_primary()]

        while self.current_token().type == TokenType.PIPE:
            self.advance()  # Skip |
            stages.append(self.parse_primary())

        if len(stages) == 1:
            return stages[0]

        return Pipeline(stages=stages, line=stages[0].line, column=stages[0].column)

    def parse_primary(self) -> Expression:
        """Parse primary expression."""
        token = self.current_token()

        # Literal string
        if token.type == TokenType.STRING:
            self.advance()
            return Literal(value=token.value, line=token.line, column=token.column)

        # Literal number
        if token.type == TokenType.NUMBER:
            self.advance()
            return Literal(value=token.value, line=token.line, column=token.column)

        # Variable
        if token.type == TokenType.VARIABLE:
            self.advance()
            return Variable(name=token.value, line=token.line, column=token.column)

        # Function call or command
        if token.type in (
            TokenType.IDENTIFIER,
            TokenType.LOAD,
            TokenType.FILTER,
            TokenType.MEASURE,
            TokenType.PLOT,
            TokenType.EXPORT,
            TokenType.GLOB,
        ):
            name = token.value
            self.advance()

            # Function call with parens
            if self.current_token().type == TokenType.LPAREN:
                return self.parse_function_call(name, token)

            # Command with args
            args = []
            while self.current_token().type not in (
                TokenType.PIPE,
                TokenType.NEWLINE,
                TokenType.EOF,
                TokenType.COLON,
                TokenType.INDENT,
                TokenType.DEDENT,
            ):
                args.append(self.parse_primary())

            return Command(name=name, args=args, line=token.line, column=token.column)

        raise SyntaxError(
            f"Unexpected token {token.type.name} at line {token.line}, column {token.column}"
        )

    def parse_function_call(self, name: str, token: Token) -> FunctionCall:
        """Parse function call with parentheses."""
        self.expect(TokenType.LPAREN)

        args = []
        while self.current_token().type != TokenType.RPAREN:
            args.append(self.parse_primary())
            if self.current_token().type == TokenType.COMMA:
                self.advance()

        self.expect(TokenType.RPAREN)
        return FunctionCall(name=name, args=args, line=token.line, column=token.column)

    def parse_for_loop(self) -> ForLoop:
        """Parse for loop with indented body.

        Supports both single-line body and multi-line indented blocks:

        Single line:
            for $f in glob("*.wfm"): load $f

        Multi-line (indented block):
            for $f in glob("*.wfm"):
                $data = load $f
                measure $data
                plot $data

        Returns:
            ForLoop AST node.
        """
        token = self.current_token()
        self.expect(TokenType.FOR)

        var_token = self.expect(TokenType.VARIABLE)
        self.expect(TokenType.IN)

        iterable = self.parse_primary()
        self.expect(TokenType.COLON)

        body: list[Statement] = []

        # Check if body follows on same line or is indented block
        if self.current_token().type == TokenType.NEWLINE:
            # Multi-line block: expect INDENT, statements, DEDENT
            self.skip_newlines()

            if self.current_token().type == TokenType.INDENT:
                self.advance()  # Consume INDENT

                # Parse statements until DEDENT
                while self.current_token().type not in (TokenType.DEDENT, TokenType.EOF):
                    self.skip_newlines()
                    if self.current_token().type in (TokenType.DEDENT, TokenType.EOF):
                        break
                    stmt = self.parse_statement()
                    body.append(stmt)
                    self.skip_newlines()

                # Consume DEDENT if present
                if self.current_token().type == TokenType.DEDENT:
                    self.advance()
            else:
                # No INDENT after newline - parse single statement
                body = [self.parse_statement()]
        else:
            # Single-line body (statement on same line as colon)
            body = [self.parse_statement()]

        return ForLoop(
            variable=var_token.value,
            iterable=iterable,
            body=body,
            line=token.line,
            column=token.column,
        )


def parse_dsl(source: str) -> list[Statement]:
    """Parse Oscura DSL source code.

    Args:
        source: DSL source code

    Returns:
        Abstract syntax tree (list of statements)

    Example:
        >>> # Single-line for loop
        >>> ast = parse_dsl('for $f in glob("*.wfm"): load $f')

        >>> # Multi-line indented block
        >>> ast = parse_dsl('''
        ... for $f in glob("*.wfm"):
        ...     $data = load $f
        ...     measure $data
        ... ''')

    Note:
        May raise SyntaxError on parse errors via tokenize() or parse().
    """
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()
