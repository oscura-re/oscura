#!/usr/bin/env python3
"""Generate comprehensive test templates for untested source files.

This script analyzes a Python source file and generates a comprehensive test
template with tests for all public functions, classes, and methods.
"""

import ast
from pathlib import Path


class CodeAnalyzer(ast.NodeVisitor):
    """Extract functions and classes from Python source code.

    Attributes:
        functions: List of function definitions found
        classes: List of class definitions found (with their methods)
    """

    def __init__(self) -> None:
        """Initialize code analyzer."""
        self.functions: list[ast.FunctionDef] = []
        self.classes: list[tuple[ast.ClassDef, list[ast.FunctionDef]]] = []
        self._current_class: ast.ClassDef | None = None
        self._class_methods: list[ast.FunctionDef] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition.

        Args:
            node: Function definition AST node
        """
        if self._current_class is None:
            # Module-level function
            if not node.name.startswith("_"):
                self.functions.append(node)
        else:
            # Class method
            if not node.name.startswith("_") or node.name == "__init__":
                self._class_methods.append(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition.

        Args:
            node: Class definition AST node
        """
        if not node.name.startswith("_"):
            self._current_class = node
            self._class_methods = []
            self.generic_visit(node)
            self.classes.append((node, self._class_methods))
            self._current_class = None
            self._class_methods = []


def analyze_source_file(source_path: Path) -> CodeAnalyzer:
    """Analyze Python source file to extract testable elements.

    Args:
        source_path: Path to source file

    Returns:
        CodeAnalyzer with extracted functions and classes

    Example:
        >>> analyzer = analyze_source_file(Path("src/oscura/loaders/csv.py"))
        >>> print(f"Functions: {len(analyzer.functions)}")
    """
    source_code = source_path.read_text()
    tree = ast.parse(source_code)
    analyzer = CodeAnalyzer()
    analyzer.visit(tree)
    return analyzer


def generate_test_template(source_path: Path, output_path: Path) -> str:
    """Generate comprehensive test template for a source file.

    Args:
        source_path: Path to source file to test
        output_path: Path where test file will be created

    Returns:
        Generated test template as string

    Example:
        >>> template = generate_test_template(
        ...     Path("src/oscura/loaders/csv.py"),
        ...     Path("tests/unit/loaders/test_csv.py")
        ... )
        >>> print(template[:100])
    """
    analyzer = analyze_source_file(source_path)

    # Calculate import path
    rel_path = source_path.relative_to(Path.cwd() / "src")
    module_path = str(rel_path.with_suffix("")).replace("/", ".")

    # Generate template
    lines = [
        f'"""Comprehensive tests for {source_path.name}.',
        "",
        "This test module provides complete coverage for all public functions,",
        "classes, and methods, including edge cases and error conditions.",
        '"""',
        "",
        "import pytest",
        "import numpy as np",
        f"from {module_path} import (",
    ]

    # Add imports for all testable items
    imports = []
    for func in analyzer.functions:
        imports.append(f"    {func.name},")
    for cls, _ in analyzer.classes:
        imports.append(f"    {cls.name},")

    if imports:
        lines.extend(imports)
        lines.append(")")
    else:
        lines.append("    # No public items found")
        lines.append(")")

    lines.extend(
        [
            "",
            "",
            "# Fixtures",
            "",
        ]
    )

    # Generate test classes for each class
    for cls, methods in analyzer.classes:
        lines.extend(
            [
                f"class Test{cls.name}:",
                f'    """Test suite for {cls.name} class.',
                "",
                "    Tests cover:",
                "    - Initialization and validation",
                "    - Normal operation with valid inputs",
                "    - Edge cases (empty, None, extremes)",
                "    - Error handling and exceptions",
                '    """',
                "",
            ]
        )

        # Generate fixture for class instance
        lines.extend(
            [
                "    @pytest.fixture",
                f"    def instance(self) -> {cls.name}:",
                f'        """Create {cls.name} instance for testing.',
                "",
                "        Returns:",
                f"            Configured {cls.name} instance",
                '        """',
                f"        return {cls.name}()",
                "",
            ]
        )

        # Generate test methods
        for method in methods:
            if method.name == "__init__":
                lines.extend(
                    [
                        "    def test_initialization(self) -> None:",
                        '        """Test successful initialization with valid parameters."""',
                        f"        obj = {cls.name}()",
                        "        assert obj is not None",
                        "        # TODO: Add specific attribute assertions",
                        "",
                        "    def test_initialization_invalid_params(self) -> None:",
                        '        """Test initialization fails with invalid parameters."""',
                        "        with pytest.raises((ValueError, TypeError)):",
                        f"            {cls.name}(None)  # TODO: Adjust invalid params",
                        "",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"    def test_{method.name}_success(self, instance: {cls.name}) -> None:",
                        f'        """Test {method.name} with valid inputs."""',
                        f"        result = instance.{method.name}()",
                        "        assert result is not None",
                        "        # TODO: Add specific assertions",
                        "",
                        f"    def test_{method.name}_edge_cases(self, instance: {cls.name}) -> None:",
                        f'        """Test {method.name} with edge cases."""',
                        "        # Test empty input",
                        "        # TODO: Implement edge case tests",
                        "        pass",
                        "",
                        f"    def test_{method.name}_error_handling(self, instance: {cls.name}) -> None:",
                        f'        """Test {method.name} error handling."""',
                        "        with pytest.raises((ValueError, TypeError)):",
                        f"            instance.{method.name}(None)  # TODO: Adjust invalid input",
                        "",
                    ]
                )

    # Generate test functions for module-level functions
    if analyzer.functions:
        lines.extend(
            [
                "",
                "# Module-level function tests",
                "",
            ]
        )

    for func in analyzer.functions:
        lines.extend(
            [
                f"def test_{func.name}_success() -> None:",
                f'    """Test {func.name} with valid inputs."""',
                f"    result = {func.name}()",
                "    assert result is not None",
                "    # TODO: Add specific assertions",
                "",
                f"def test_{func.name}_edge_cases() -> None:",
                f'    """Test {func.name} with edge cases."""',
                "    # Test empty input",
                "    # Test None input",
                "    # Test extreme values",
                "    pass  # TODO: Implement edge case tests",
                "",
                f"def test_{func.name}_error_handling() -> None:",
                f'    """Test {func.name} error handling."""',
                "    with pytest.raises((ValueError, TypeError)):",
                f"        {func.name}(None)  # TODO: Adjust invalid input",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    """Generate test templates for all untested files."""
    import sys

    if len(sys.argv) != 3:
        print("Usage: python generate_test_template.py <source_file> <test_file>")
        sys.exit(1)

    source_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not source_path.exists():
        print(f"Error: Source file not found: {source_path}")
        sys.exit(1)

    # Generate template
    template = generate_test_template(source_path, output_path)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write template
    output_path.write_text(template)
    print(f"Generated test template: {output_path}")
    print(f"  - {template.count('def test_')} test functions/methods")


if __name__ == "__main__":
    main()
