#!/usr/bin/env python3
"""Comprehensive audit of all example files.

Extracts functionality, APIs used, file size, complexity, and SKIP_VALIDATION status
for every example file in both demonstrations/ and demos/ directories.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

# Root directory
ROOT = Path(__file__).parent.parent


class ExampleAnalyzer:
    """Analyzes example files for comprehensive audit."""

    def __init__(self, file_path: Path):
        """Initialize analyzer.

        Args:
            file_path: Path to example file
        """
        self.path = file_path
        self.relative_path = str(file_path.relative_to(ROOT))
        self.content = file_path.read_text()
        self.size_bytes = len(self.content)
        self.lines = self.content.split("\n")

    def analyze(self) -> dict[str, Any]:
        """Perform comprehensive analysis.

        Returns:
            Dictionary with analysis results
        """
        return {
            "path": self.relative_path,
            "directory": self.path.parent.name,
            "filename": self.path.name,
            "size_bytes": self.size_bytes,
            "line_count": len(self.lines),
            "docstring": self._extract_docstring(),
            "imports": self._extract_imports(),
            "oscura_apis": self._extract_oscura_apis(),
            "skip_validation": self._check_skip_validation(),
            "skip_reason": self._get_skip_reason(),
            "has_main": self._has_main(),
            "complexity_score": self._estimate_complexity(),
            "demonstrates": self._extract_demonstrates(),
            "related_demos": self._extract_related_demos(),
            "standards": self._extract_standards(),
        }

    def _extract_docstring(self) -> str | None:
        """Extract module-level docstring."""
        try:
            tree = ast.parse(self.content)
            docstring = ast.get_docstring(tree)
            return docstring.split("\n")[0] if docstring else None
        except Exception:
            return None

    def _extract_imports(self) -> list[str]:
        """Extract all import statements."""
        imports = []
        try:
            tree = ast.parse(self.content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}" if module else alias.name)
        except Exception:
            pass
        return imports

    def _extract_oscura_apis(self) -> list[str]:
        """Extract Oscura API calls and imports."""
        apis = set()

        # From imports
        for imp in self._extract_imports():
            if "oscura" in imp:
                apis.add(imp)

        # From code (function calls, class usage)
        oscura_pattern = re.compile(r"\boscura\.(\w+(?:\.\w+)*)", re.MULTILINE)
        for match in oscura_pattern.finditer(self.content):
            apis.add(f"oscura.{match.group(1)}")

        return sorted(apis)

    def _check_skip_validation(self) -> bool:
        """Check if file has SKIP_VALIDATION marker."""
        return "# SKIP_VALIDATION" in self.content[:2000]  # Check first 2000 chars

    def _get_skip_reason(self) -> str | None:
        """Extract SKIP_VALIDATION reason if present."""
        for line in self.lines[:50]:
            if "# SKIP_VALIDATION" in line and ":" in line:
                return line.split(":", 1)[1].strip()
        return None

    def _has_main(self) -> bool:
        """Check if file has __main__ block."""
        return '__name__ == "__main__"' in self.content

    def _estimate_complexity(self) -> int:
        """Estimate code complexity (simple metric).

        Returns:
            Complexity score (higher = more complex)
        """
        try:
            tree = ast.parse(self.content)

            # Count various complexity indicators
            functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            loops = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While)))
            conditionals = sum(1 for node in ast.walk(tree) if isinstance(node, ast.If))

            # Weighted complexity score
            return functions * 2 + classes * 5 + loops + conditionals

        except Exception:
            # Fallback: just count lines
            return len(self.lines) // 10

    def _extract_demonstrates(self) -> list[str]:
        """Extract 'Demonstrates:' section from docstring."""
        demonstrates = []
        in_demonstrates = False

        for line in self.lines[:100]:  # Check first 100 lines
            if "Demonstrates:" in line or "demonstrates:" in line.lower():
                in_demonstrates = True
                continue

            if in_demonstrates:
                line = line.strip()
                if not line:
                    break  # End of demonstrates section
                if line.startswith("-"):
                    demonstrates.append(line.lstrip("- ").strip())
                elif line.startswith("*"):
                    demonstrates.append(line.lstrip("* ").strip())
                elif ":" in line:
                    break  # Next section

        return demonstrates

    def _extract_related_demos(self) -> list[str]:
        """Extract related demo references."""
        related = []
        for line in self.lines[:100]:
            if "Related" in line and ":" in line:
                # Extract file references
                refs = re.findall(r"(\d+_\w+/\d+_[\w_]+\.py)", line)
                related.extend(refs)
        return related

    def _extract_standards(self) -> list[str]:
        """Extract IEEE/ISO standards mentioned."""
        standards = []
        standards_pattern = re.compile(r"\b(IEEE|ISO|CISPR|FCC|MIL-STD)\s*[-\s]*\d+[-\d]*\b")
        for match in standards_pattern.finditer(self.content):
            standards.append(match.group(0))
        return list(set(standards))


def main() -> None:
    """Main entry point."""

    # Find all Python files in both directories
    demonstrations = sorted((ROOT / "demonstrations").rglob("*.py"))
    demos = sorted((ROOT / "demos").rglob("*.py"))

    # Filter out utility files
    exclude_names = {
        "__init__.py",
        "validate_all.py",
        "capability_index.py",
        "generate_all_data.py",
        "generate_all_demo_data.py",
        "comprehensive_demo_checker.py",
        "validate_all_demos.py",
    }

    demonstrations = [
        f for f in demonstrations if f.name not in exclude_names and "common" not in f.parts
    ]
    demos = [
        f
        for f in demos
        if f.name not in exclude_names
        and "common" not in f.parts
        and "data_generation" not in f.parts
    ]

    print(f"Analyzing {len(demonstrations)} files in demonstrations/")
    print(f"Analyzing {len(demos)} files in demos/")
    print()

    # Analyze all files
    all_analyses = []

    for file_path in demonstrations + demos:
        try:
            analyzer = ExampleAnalyzer(file_path)
            analysis = analyzer.analyze()
            all_analyses.append(analysis)

            # Show progress
            directory = "demonstrations" if "demonstrations" in str(file_path) else "demos"
            skip_marker = " [SKIP]" if analysis["skip_validation"] else ""
            print(f"  ✓ {directory}/{analysis['directory']}/{analysis['filename']}{skip_marker}")

        except Exception as e:
            print(f"  ✗ Error analyzing {file_path}: {e}")

    # Write comprehensive report
    output_file = ROOT / ".claude" / "examples_audit_comprehensive.json"
    with open(output_file, "w") as f:
        json.dump(all_analyses, f, indent=2)

    print()
    print(f"✓ Analysis complete: {output_file}")
    print()

    # Generate summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()

    demonstrations_count = len([a for a in all_analyses if "demonstrations/" in a["path"]])
    demos_count = len([a for a in all_analyses if "demos/" in a["path"]])

    print(f"Total files analyzed: {len(all_analyses)}")
    print(f"  - demonstrations/: {demonstrations_count}")
    print(f"  - demos/: {demos_count}")
    print()

    skip_count = len([a for a in all_analyses if a["skip_validation"]])
    print(f"Files with SKIP_VALIDATION: {skip_count}")

    # Total lines and size
    total_lines = sum(a["line_count"] for a in all_analyses)
    total_size = sum(a["size_bytes"] for a in all_analyses)
    print(f"Total lines of code: {total_lines:,}")
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    print()

    # Complexity distribution
    avg_complexity = sum(a["complexity_score"] for a in all_analyses) / len(all_analyses)
    print(f"Average complexity score: {avg_complexity:.1f}")
    print()

    # Extract all unique Oscura APIs
    all_apis = set()
    for analysis in all_analyses:
        all_apis.update(analysis["oscura_apis"])

    print(f"Unique Oscura APIs used: {len(all_apis)}")
    print()

    # Categories in each directory
    demo_categories = set(a["directory"] for a in all_analyses if "demonstrations/" in a["path"])
    demos_categories = set(a["directory"] for a in all_analyses if "demos/" in a["path"])

    print(f"Categories in demonstrations/: {len(demo_categories)}")
    print(f"Categories in demos/: {len(demos_categories)}")
    print()

    # Potential duplicates (same category name in both)
    overlapping = demo_categories & demos_categories
    if overlapping:
        print(f"Overlapping categories: {len(overlapping)}")
        for cat in sorted(overlapping):
            print(f"  - {cat}")
        print()

    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review examples_audit_comprehensive.json for detailed analysis")
    print("2. Identify functionality-based duplicates")
    print("3. Design optimal category structure")
    print()


if __name__ == "__main__":
    main()
