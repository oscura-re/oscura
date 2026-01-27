#!/usr/bin/env python3
"""Automated refactoring tool for functions exceeding complexity/length thresholds.

This script systematically identifies and refactors long functions (>100 lines)
and complex functions (complexity >15) across the entire codebase.

Usage:
    python3 scripts/refactor_long_functions.py --analyze   # Analyze only
    python3 scripts/refactor_long_functions.py --refactor --batch 10  # Refactor in batches
    python3 scripts/refactor_long_functions.py --file src/oscura/config/schema.py  # Single file

Strategy:
    - Extract logical blocks into helper methods
    - Extract validation logic into _validate_*() methods
    - Extract processing logic into _process_*() methods
    - Target: Functions <100 lines, complexity <15
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FunctionInfo:
    """Information about a function requiring refactoring.

    Attributes:
        filepath: Path to the file containing the function.
        name: Function name.
        lineno: Starting line number.
        end_lineno: Ending line number.
        lines: Total line count.
        complexity: Cyclomatic complexity score.
        ast_node: AST node for the function.
    """

    filepath: Path
    name: str
    lineno: int
    end_lineno: int
    lines: int
    complexity: int
    ast_node: ast.FunctionDef


class FunctionAnalyzer:
    """Analyze Python source code for long and complex functions."""

    def __init__(self, line_threshold: int = 100, complexity_threshold: int = 15) -> None:
        """Initialize analyzer with configurable thresholds.

        Args:
            line_threshold: Maximum acceptable function length.
            complexity_threshold: Maximum acceptable cyclomatic complexity.
        """
        self.line_threshold = line_threshold
        self.complexity_threshold = complexity_threshold

    def analyze_file(self, filepath: Path) -> list[FunctionInfo]:
        """Analyze a single Python file for long/complex functions.

        Args:
            filepath: Path to Python file to analyze.

        Returns:
            List of FunctionInfo objects for functions exceeding thresholds.
        """
        try:
            source = filepath.read_text()
            tree = ast.parse(source, filename=str(filepath))
        except Exception as e:
            print(f"ERROR parsing {filepath}: {e}", file=sys.stderr)
            return []

        # Calculate complexity
        complexity_map = self._get_complexity_map(source)

        # Find long/complex functions
        problematic = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue

            lines = node.end_lineno - node.lineno + 1 if node.end_lineno else 0
            complexity = complexity_map.get(node.name, 1)

            if lines > self.line_threshold or complexity > self.complexity_threshold:
                problematic.append(
                    FunctionInfo(
                        filepath=filepath,
                        name=node.name,
                        lineno=node.lineno,
                        end_lineno=node.end_lineno or node.lineno,
                        lines=lines,
                        complexity=complexity,
                        ast_node=node,
                    )
                )

        return problematic

    def _get_complexity_map(self, source: str) -> dict[str, int]:
        """Calculate cyclomatic complexity for all functions using AST.

        Args:
            source: Python source code.

        Returns:
            Mapping of function name to complexity score.
        """
        try:
            tree = ast.parse(source)
            complexity_map = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity_map[node.name] = self._calculate_complexity(node)
            return complexity_map
        except Exception:
            return {}

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function.

        Args:
            node: Function AST node.

        Returns:
            Cyclomatic complexity score.
        """
        complexity = 1  # Base complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def analyze_directory(self, directory: Path) -> list[FunctionInfo]:
        """Recursively analyze all Python files in a directory.

        Args:
            directory: Root directory to analyze.

        Returns:
            List of FunctionInfo objects for all problematic functions.
        """
        all_problematic = []
        for pyfile in directory.rglob("*.py"):
            all_problematic.extend(self.analyze_file(pyfile))

        # Sort by complexity (highest first), then by lines
        all_problematic.sort(key=lambda f: (f.complexity, f.lines), reverse=True)
        return all_problematic


class RefactoringStrategy:
    """Strategies for refactoring long/complex functions."""

    @staticmethod
    def suggest_strategy(func_info: FunctionInfo) -> str:
        """Suggest refactoring strategy based on function characteristics.

        Args:
            func_info: Information about the function to refactor.

        Returns:
            Human-readable refactoring suggestion.
        """
        strategies = []

        # Check function type based on name
        if func_info.name.startswith("_register"):
            strategies.append("Extract each registration into separate _create_*() function")
        elif func_info.name.startswith("plot_") or func_info.name.startswith("visualize_"):
            strategies.append("Extract plot setup, data processing, and rendering into helpers")
        elif "analyze" in func_info.name or "process" in func_info.name:
            strategies.append(
                "Extract validation, processing, and aggregation into separate methods"
            )
        elif func_info.name == "__init__":
            strategies.append("Extract initialization logic into _setup_*() methods")

        # Generic strategies based on complexity
        if func_info.complexity > 30:
            strategies.append(
                "High complexity - extract conditional branches into separate methods"
            )
        if func_info.lines > 200:
            strategies.append("Very long - split into logical phases (validate, process, format)")

        # Always applicable
        strategies.append("Extract validation into _validate_*() methods")
        strategies.append("Extract data processing into _process_*() methods")

        return " | ".join(strategies) if strategies else "General refactoring needed"


def print_analysis_report(functions: list[FunctionInfo]) -> None:
    """Print formatted analysis report to stdout.

    Args:
        functions: List of problematic functions to report.
    """
    print("\n" + "=" * 100)
    print(f"REFACTORING ANALYSIS - {len(functions)} functions require refactoring")
    print("=" * 100 + "\n")

    # Group by file
    by_file: dict[Path, list[FunctionInfo]] = {}
    for func in functions:
        by_file.setdefault(func.filepath, []).append(func)

    print(f"Affected files: {len(by_file)}\n")

    # Print top 20 worst offenders
    print("Top 20 Functions by Complexity:")
    print(f"{'Lines':>6} | {'Cmplx':>6} | {'File:Line':<60} | {'Function':<30}")
    print("-" * 100)

    for func in functions[:20]:
        try:
            filepath_display = str(func.filepath.relative_to(Path.cwd()))
        except ValueError:
            filepath_display = str(func.filepath)
        location = f"{filepath_display}:{func.lineno}"
        print(f"{func.lines:6d} | {func.complexity:6d} | {location:<60} | {func.name:<30}")

    print("\n" + "=" * 100)
    print(f"\nTotal functions to refactor: {len(functions)}")
    print(f"Estimated effort: {len(functions) * 15} minutes ({len(functions) * 15 / 60:.1f} hours)")
    print("\nRun with --refactor to start automated refactoring")


def main() -> int:
    """Main entry point for refactoring tool.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        description="Automated refactoring tool for long and complex functions"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze codebase and report problematic functions",
    )
    parser.add_argument(
        "--refactor",
        action="store_true",
        help="Perform automated refactoring (use with caution)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Analyze or refactor specific file only",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=10,
        help="Number of functions to refactor per batch (default: 10)",
    )
    parser.add_argument(
        "--line-threshold",
        type=int,
        default=100,
        help="Maximum acceptable function length (default: 100)",
    )
    parser.add_argument(
        "--complexity-threshold",
        type=int,
        default=15,
        help="Maximum acceptable cyclomatic complexity (default: 15)",
    )

    args = parser.parse_args()

    analyzer = FunctionAnalyzer(
        line_threshold=args.line_threshold,
        complexity_threshold=args.complexity_threshold,
    )

    # Determine what to analyze
    if args.file:
        if not args.file.exists():
            print(f"ERROR: File not found: {args.file}", file=sys.stderr)
            return 1
        functions = analyzer.analyze_file(args.file)
    else:
        src_dir = Path("src/oscura")
        if not src_dir.exists():
            print(f"ERROR: Source directory not found: {src_dir}", file=sys.stderr)
            return 1
        functions = analyzer.analyze_directory(src_dir)

    if not functions:
        print("No functions found exceeding thresholds - codebase is clean!")
        return 0

    # Print analysis report
    print_analysis_report(functions)

    # Refactoring is complex and requires careful AST manipulation
    # For now, this tool focuses on analysis and reporting
    if args.refactor:
        print("\n" + "=" * 100)
        print("REFACTORING MODE")
        print("=" * 100)
        print("\nAutomated refactoring is a complex task requiring:")
        print("  1. AST-based code transformation")
        print("  2. Test validation after each change")
        print("  3. Git commits per batch")
        print("\nRecommended approach:")
        print("  1. Use this analysis to identify priority files")
        print("  2. Manually refactor using IDE + code_assistant agent")
        print("  3. Run ./scripts/test.sh after each file")
        print("  4. Commit working changes before moving to next file")
        print("\nFor automated assistance, use:")
        print(
            "  /route code_assistant 'Refactor src/oscura/config/schema.py:_register_builtin_schemas'"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
