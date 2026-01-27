#!/usr/bin/env python3
"""Complete complexity analysis - identify ALL functions exceeding thresholds."""

import ast
import json
from pathlib import Path
from typing import Any, Union


class ComplexityAnalyzer(ast.NodeVisitor):
    """Analyze cyclomatic complexity and line count for functions."""

    def __init__(self) -> None:
        """Initialize analyzer."""
        self.functions: list[dict[str, Any]] = []
        self.current_file = ""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition node."""
        complexity = self._calculate_complexity(node)
        line_count = node.end_lineno - node.lineno + 1 if node.end_lineno else 0

        self.functions.append(
            {
                "name": node.name,
                "file": self.current_file,
                "lineno": node.lineno,
                "lines": line_count,
                "complexity": complexity,
                "exceeds_lines": line_count > 100,
                "exceeds_complexity": complexity > 15,
            }
        )
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition node."""
        complexity = self._calculate_complexity(node)
        line_count = node.end_lineno - node.lineno + 1 if node.end_lineno else 0

        self.functions.append(
            {
                "name": node.name,
                "file": self.current_file,
                "lineno": node.lineno,
                "lines": line_count,
                "complexity": complexity,
                "exceeds_lines": line_count > 100,
                "exceeds_complexity": complexity > 15,
            }
        )
        self.generic_visit(node)

    def _calculate_complexity(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(
                child,
                (
                    ast.If,
                    ast.While,
                    ast.For,
                    ast.AsyncFor,
                    ast.ExceptHandler,
                    ast.With,
                    ast.AsyncWith,
                ),
            ):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                if child.ifs:
                    complexity += len(child.ifs)
        return complexity


def analyze_file(file_path: Path) -> list[dict[str, Any]]:
    """Analyze a single Python file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        analyzer = ComplexityAnalyzer()
        analyzer.current_file = str(file_path)
        analyzer.visit(tree)
        return analyzer.functions
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return []


def main() -> None:
    """Run complete complexity analysis."""
    src_dir = Path("/home/lair-click-bats/development/oscura/src")
    all_functions = []

    # Analyze all Python files
    for py_file in sorted(src_dir.rglob("*.py")):
        if ".venv" not in str(py_file):
            functions = analyze_file(py_file)
            all_functions.extend(functions)

    # Filter functions exceeding thresholds
    problematic = [f for f in all_functions if f["exceeds_lines"] or f["exceeds_complexity"]]

    # Sort by complexity first, then lines
    problematic.sort(key=lambda x: (x["complexity"], x["lines"]), reverse=True)

    # Create report
    report = {
        "total_functions": len(all_functions),
        "problematic_functions": len(problematic),
        "exceeding_lines": len([f for f in problematic if f["exceeds_lines"]]),
        "exceeding_complexity": len([f for f in problematic if f["exceeds_complexity"]]),
        "functions": problematic,
    }

    # Print summary
    print(f"\n{'=' * 80}")
    print("COMPLETE COMPLEXITY ANALYSIS")
    print(f"{'=' * 80}")
    print(f"Total functions analyzed: {report['total_functions']}")
    print(f"Functions exceeding thresholds: {report['problematic_functions']}")
    print(f"  - Exceeding 100 lines: {report['exceeding_lines']}")
    print(f"  - Exceeding complexity 15: {report['exceeding_complexity']}")
    print(f"{'=' * 80}\n")

    # Print top 50
    print("TOP 50 FUNCTIONS BY COMPLEXITY:\n")
    for i, func in enumerate(problematic[:50], 1):
        relative_path = func["file"].replace("/home/lair-click-bats/development/oscura/", "")
        markers = []
        if func["exceeds_complexity"]:
            markers.append(f"C={func['complexity']}")
        if func["exceeds_lines"]:
            markers.append(f"L={func['lines']}")
        status = " ".join(markers)
        print(f"{i:3}. [{status:20}] {relative_path}:{func['lineno']} - {func['name']}")

    # Save to JSON
    output_path = Path("/home/lair-click-bats/development/oscura/.claude/agent-outputs")
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "complexity_analysis_complete.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n\nFull report saved to: {output_file}")
    print(f"Total problematic functions: {len(problematic)}")


if __name__ == "__main__":
    main()
