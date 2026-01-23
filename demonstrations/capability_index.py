#!/usr/bin/env python3
"""Generate comprehensive capability index for all demonstrations.

Analyzes all demonstrations and generates:
- API coverage matrix (which demos cover which capabilities)
- Module coverage report
- Gap analysis (capabilities without demonstrations)
- Cross-reference index

Usage:
    python demonstrations/capability_index.py
    python demonstrations/capability_index.py --format markdown  # Output to INDEX.md
    python demonstrations/capability_index.py --gaps-only  # Only show coverage gaps
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path


class CapabilityExtractor(ast.NodeVisitor):
    """Extract demonstrated capabilities from Python source."""

    def __init__(self):
        """Initialize extractor."""
        self.imports: set[str] = set()
        self.calls: set[str] = set()

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Extract from imports."""
        if node.module and node.module.startswith("oscura"):
            for alias in node.names:
                self.imports.add(f"{node.module}.{alias.name}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Extract function calls."""
        if isinstance(node.func, ast.Attribute):
            # Handle osc.function() calls
            if isinstance(node.func.value, ast.Name):
                self.calls.add(f"{node.func.value.id}.{node.func.attr}")
        elif isinstance(node.func, ast.Name):
            # Handle direct function() calls
            self.calls.add(node.func.id)
        self.generic_visit(node)


class CapabilityIndexer:
    """Generate comprehensive capability index."""

    def __init__(self):
        """Initialize indexer."""
        self.demos_dir = Path(__file__).parent
        self.demo_capabilities: dict[str, set[str]] = {}
        self.capability_demos: dict[str, set[str]] = defaultdict(set)
        self.all_api_symbols: set[str] = set()

    def discover_demos(self) -> list[Path]:
        """Discover all demonstration files.

        Returns:
            List of demonstration file paths
        """
        demo_files = sorted(self.demos_dir.glob("**/*.py"))

        # Exclude common/ and utility scripts
        demo_files = [
            f
            for f in demo_files
            if f.parent.name != "common"
            and f.name not in ["validate_all.py", "capability_index.py", "generate_all_data.py"]
        ]

        return demo_files

    def extract_capabilities(self, demo_path: Path) -> set[str]:
        """Extract capabilities demonstrated in a file.

        Args:
            demo_path: Path to demonstration file

        Returns:
            Set of capabilities (e.g., {"oscura.fft", "oscura.thd"})
        """
        try:
            source = demo_path.read_text()
            tree = ast.parse(source)
            extractor = CapabilityExtractor()
            extractor.visit(tree)

            # Combine imports and calls
            capabilities = extractor.imports | extractor.calls
            return {cap for cap in capabilities if "oscura" in cap.lower()}

        except Exception as e:
            print(f"Warning: Could not parse {demo_path}: {e}", file=sys.stderr)
            return set()

    def load_api_symbols(self) -> set[str]:
        """Load all exported API symbols from oscura.__init__.py.

        Returns:
            Set of all API symbols
        """
        try:
            init_file = self.demos_dir.parent / "src" / "oscura" / "__init__.py"
            if not init_file.exists():
                print("Warning: Could not find oscura/__init__.py", file=sys.stderr)
                return set()

            source = init_file.read_text()

            # Extract __all__ list
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "__all__":
                            if isinstance(node.value, ast.List):
                                return {
                                    elt.s for elt in node.value.elts if isinstance(elt, ast.Str)
                                }

            return set()

        except Exception as e:
            print(f"Warning: Could not load API symbols: {e}", file=sys.stderr)
            return set()

    def analyze_coverage(self) -> None:
        """Analyze demonstration coverage of API."""
        demos = self.discover_demos()
        self.all_api_symbols = self.load_api_symbols()

        print(f"Analyzing {len(demos)} demonstrations...")
        print(f"Tracking {len(self.all_api_symbols)} API symbols...")

        for demo_path in demos:
            relative_path = str(demo_path.relative_to(self.demos_dir))
            capabilities = self.extract_capabilities(demo_path)

            self.demo_capabilities[relative_path] = capabilities

            # Build reverse index
            for cap in capabilities:
                self.capability_demos[cap].add(relative_path)

    def generate_report(self, format: str = "text", gaps_only: bool = False) -> str:
        """Generate capability index report.

        Args:
            format: Output format ("text" or "markdown")
            gaps_only: Only show coverage gaps

        Returns:
            Formatted report
        """
        lines = []

        if format == "markdown":
            lines.append("# Oscura Demonstration Capability Index")
            lines.append("")
            lines.append(f"**Generated:** {__import__('datetime').datetime.now().isoformat()}")
            lines.append("")
        else:
            lines.append("=" * 80)
            lines.append("OSCURA DEMONSTRATION CAPABILITY INDEX")
            lines.append("=" * 80)
            lines.append("")

        # Summary statistics
        total_demos = len(self.demo_capabilities)
        total_capabilities = len(self.capability_demos)
        demonstrated_api_symbols = sum(
            1
            for symbol in self.all_api_symbols
            if any(symbol in cap for cap in self.capability_demos)
        )

        coverage_pct = (
            demonstrated_api_symbols / len(self.all_api_symbols) * 100
            if self.all_api_symbols
            else 0
        )

        if format == "markdown":
            lines.append("## Summary")
            lines.append("")
            lines.append(f"- **Total Demonstrations:** {total_demos}")
            lines.append(f"- **Capabilities Demonstrated:** {total_capabilities}")
            lines.append(f"- **API Symbols in __all__:** {len(self.all_api_symbols)}")
            lines.append(f"- **API Symbols Demonstrated:** {demonstrated_api_symbols}")
            lines.append(f"- **API Coverage:** {coverage_pct:.1f}%")
            lines.append("")
        else:
            lines.append("SUMMARY")
            lines.append("-" * 80)
            lines.append(f"Total Demonstrations: {total_demos}")
            lines.append(f"Capabilities Demonstrated: {total_capabilities}")
            lines.append(f"API Symbols in __all__: {len(self.all_api_symbols)}")
            lines.append(f"API Symbols Demonstrated: {demonstrated_api_symbols}")
            lines.append(f"API Coverage: {coverage_pct:.1f}%")
            lines.append("")

        if not gaps_only:
            # Demonstrations by section
            demos_by_section: dict[str, list[str]] = defaultdict(list)
            for demo_path in self.demo_capabilities:
                section = demo_path.split("/")[0]
                demos_by_section[section].append(demo_path)

            if format == "markdown":
                lines.append("## Demonstrations by Section")
                lines.append("")
                for section in sorted(demos_by_section.keys()):
                    lines.append(f"### {section}")
                    lines.append("")
                    for demo in sorted(demos_by_section[section]):
                        cap_count = len(self.demo_capabilities[demo])
                        lines.append(f"- **{demo}** ({cap_count} capabilities)")
                    lines.append("")
            else:
                lines.append("DEMONSTRATIONS BY SECTION")
                lines.append("-" * 80)
                for section in sorted(demos_by_section.keys()):
                    lines.append(f"\n{section}:")
                    for demo in sorted(demos_by_section[section]):
                        cap_count = len(self.demo_capabilities[demo])
                        lines.append(f"  - {demo} ({cap_count} capabilities)")
                lines.append("")

        # Coverage gaps
        if format == "markdown":
            lines.append("## Coverage Gaps")
            lines.append("")
        else:
            lines.append("COVERAGE GAPS")
            lines.append("-" * 80)

        gaps = [
            symbol
            for symbol in sorted(self.all_api_symbols)
            if not any(symbol in cap for cap in self.capability_demos)
        ]

        if gaps:
            lines.append(f"Found {len(gaps)} API symbols without demonstrations:")
            lines.append("")
            if format == "markdown":
                for gap in gaps:
                    lines.append(f"- `{gap}`")
            else:
                for gap in gaps:
                    lines.append(f"  - {gap}")
        else:
            lines.append("✓ 100% API coverage - all symbols demonstrated!")

        lines.append("")

        return "\n".join(lines)

    def print_report(self, format: str = "text", gaps_only: bool = False) -> None:
        """Print capability index report.

        Args:
            format: Output format
            gaps_only: Only show gaps
        """
        report = self.generate_report(format=format, gaps_only=gaps_only)
        print(report)

    def save_report(self, output_path: Path, format: str = "markdown") -> None:
        """Save report to file.

        Args:
            output_path: Path to save report
            format: Output format
        """
        report = self.generate_report(format=format)
        output_path.write_text(report)
        print(f"Report saved to: {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate demonstration capability index")
    parser.add_argument(
        "--format",
        choices=["text", "markdown"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--output", "-o", help="Save report to file (auto-detects format from extension)"
    )
    parser.add_argument("--gaps-only", action="store_true", help="Only show coverage gaps")

    args = parser.parse_args()

    indexer = CapabilityIndexer()
    indexer.analyze_coverage()

    if args.output:
        output_path = Path(args.output)
        format_arg = "markdown" if output_path.suffix == ".md" else "text"
        indexer.save_report(output_path, format=format_arg)
    else:
        indexer.print_report(format=args.format, gaps_only=args.gaps_only)

    return 0


if __name__ == "__main__":
    sys.exit(main())
