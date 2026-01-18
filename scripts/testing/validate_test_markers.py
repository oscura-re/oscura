#!/usr/bin/env python3
"""Validate and auto-fix pytest markers in test files.

This script ensures all test files have proper module-level pytestmark declarations.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple


class MarkerInfo(NamedTuple):
    """Information about a test file's markers."""

    file_path: Path
    has_pytestmark: bool
    has_inline_markers: bool
    suggested_markers: list[str]
    line_with_pytestmark: int | None


def determine_markers_for_file(file_path: Path) -> list[str]:
    """Determine appropriate markers based on file path.

    Args:
        file_path: Path to test file

    Returns:
        List of marker strings (e.g., ['pytest.mark.unit', 'pytest.mark.loader'])
    """
    parts = file_path.parts
    markers = []

    # Test level markers
    if "unit" in parts:
        markers.append("pytest.mark.unit")
    elif "integration" in parts:
        markers.append("pytest.mark.integration")
    elif "performance" in parts:
        markers.append("pytest.mark.performance")
    elif "compliance" in parts:
        markers.append("pytest.mark.compliance")
    elif "stress" in parts:
        markers.append("pytest.mark.stress")
    elif "regression" in parts:
        markers.append("pytest.mark.regression")

    # Domain markers - top level
    if "loaders" in parts:
        markers.append("pytest.mark.loader")
    elif "analyzers" in parts:
        markers.append("pytest.mark.analyzer")
    elif "inference" in parts:
        markers.append("pytest.mark.inference")
    elif "exporters" in parts:
        markers.append("pytest.mark.exporter")
    elif "core" in parts:
        markers.append("pytest.mark.core")
    elif "cli" in parts:
        markers.append("pytest.mark.cli")
    elif "visualization" in parts:
        markers.append("pytest.mark.visualization")

    # Subdomain markers - analyzers
    if "digital" in parts:
        markers.append("pytest.mark.digital")
    elif "spectral" in parts:
        markers.append("pytest.mark.spectral")
    elif "statistical" in parts:
        markers.append("pytest.mark.statistical")
    elif "protocol" in parts:
        markers.append("pytest.mark.protocol")
    elif "patterns" in parts:
        markers.append("pytest.mark.pattern")
    elif "power" in parts:
        markers.append("pytest.mark.power")
    elif "jitter" in parts:
        markers.append("pytest.mark.jitter")
    elif "eye" in parts:
        markers.append("pytest.mark.eye")
    elif "packet" in parts:
        markers.append("pytest.mark.packet")

    # Workflow markers (integration tests)
    if "integration" in parts and "workflow" in file_path.name.lower():
        markers.append("pytest.mark.workflow")

    return markers


def analyze_file(file_path: Path) -> MarkerInfo:
    """Analyze a test file for marker usage.

    Args:
        file_path: Path to test file

    Returns:
        MarkerInfo with analysis results
    """
    content = file_path.read_text()
    lines = content.splitlines()

    has_pytestmark = False
    pytestmark_line = None
    has_inline_markers = False

    for i, line in enumerate(lines, 1):
        if re.match(r"^pytestmark\s*=", line):
            has_pytestmark = True
            pytestmark_line = i
        if re.search(r"@pytest\.mark\.", line):
            has_inline_markers = True

    suggested_markers = determine_markers_for_file(file_path)

    return MarkerInfo(
        file_path=file_path,
        has_pytestmark=has_pytestmark,
        has_inline_markers=has_inline_markers,
        suggested_markers=suggested_markers,
        line_with_pytestmark=pytestmark_line,
    )


def fix_file(marker_info: MarkerInfo, remove_redundant: bool = False) -> bool:
    """Add pytestmark to a file missing it.

    Args:
        marker_info: MarkerInfo for the file
        remove_redundant: If True, remove inline markers that duplicate pytestmark

    Returns:
        True if file was modified, False otherwise
    """
    if marker_info.has_pytestmark:
        return False

    if not marker_info.suggested_markers:
        print(f"⚠️  No markers determined for {marker_info.file_path}")
        return False

    content = marker_info.file_path.read_text()
    lines = content.splitlines(keepends=True)

    # Find insertion point (after imports, before first class/function)
    insertion_idx = 0
    in_docstring = False
    docstring_char = None
    in_multiline_import = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track docstrings
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if not in_docstring:
                in_docstring = True
                docstring_char = stripped[:3]
                if stripped.count(docstring_char) >= 2:
                    in_docstring = False
            elif docstring_char in stripped:
                in_docstring = False

        # Skip if in docstring
        if in_docstring:
            continue

        # Track multi-line imports
        if (stripped.startswith("import ") or stripped.startswith("from ")) and "(" in line:
            in_multiline_import = True
            insertion_idx = i + 1
            continue

        if in_multiline_import:
            insertion_idx = i + 1
            if ")" in line:
                in_multiline_import = False
            continue

        # Found single-line import statement
        if stripped.startswith("import ") or stripped.startswith("from "):
            insertion_idx = i + 1

        # Stop at first class or function definition
        if (
            stripped.startswith("class ")
            or stripped.startswith("def ")
            or stripped.startswith("@pytest.")
        ):
            break

    # Build pytestmark line
    if len(marker_info.suggested_markers) == 1:
        marker_line = f"pytestmark = {marker_info.suggested_markers[0]}\n"
    else:
        markers_str = ", ".join(marker_info.suggested_markers)
        marker_line = f"pytestmark = [{markers_str}]\n"

    # Insert pytestmark with blank line before and after
    lines.insert(insertion_idx, "\n")
    lines.insert(insertion_idx + 1, marker_line)
    lines.insert(insertion_idx + 2, "\n")

    # Write back
    marker_info.file_path.write_text("".join(lines))
    return True


def validate_pytestmark_format(marker_info: MarkerInfo) -> list[str]:
    """Validate that pytestmark uses correct format.

    Args:
        marker_info: MarkerInfo for the file

    Returns:
        List of validation errors (empty if valid)
    """
    if not marker_info.has_pytestmark:
        return []

    content = marker_info.file_path.read_text()
    lines = content.splitlines()

    errors = []

    for line in lines:
        if re.match(r"^pytestmark\s*=", line):
            # Check for string markers (invalid)
            if re.search(r'pytestmark\s*=\s*\[?"[^"]*"', line):
                errors.append("Uses string markers instead of pytest.mark objects")

            # Check if single marker without list (acceptable but inconsistent)
            if (
                "pytest.mark." in line
                and "[" not in line
                and "," not in line
                and len(marker_info.suggested_markers) > 1
            ):
                errors.append("Single marker without list, but multiple markers expected")

    return errors


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate and fix pytest markers in test files")
    parser.add_argument("--fix", action="store_true", help="Auto-fix files missing pytestmark")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any missing markers (for CI)",
    )
    parser.add_argument(
        "--remove-redundant",
        action="store_true",
        help="Remove inline decorators that duplicate pytestmark",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["tests"],
        help="Paths to validate (default: tests/)",
    )

    args = parser.parse_args()

    # Collect all test files
    test_files = []
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file():
            test_files.append(path)
        elif path.is_dir():
            test_files.extend(path.rglob("test_*.py"))

    if not test_files:
        print("No test files found")
        return 0

    # Analyze all files
    print(f"Analyzing {len(test_files)} test files...")
    results = [analyze_file(f) for f in test_files]

    # Report statistics
    missing_pytestmark = [r for r in results if not r.has_pytestmark]
    has_inline = [r for r in results if r.has_inline_markers]
    has_both = [r for r in results if r.has_pytestmark and r.has_inline_markers]

    print("\n📊 Statistics:")
    print(f"  Total test files: {len(results)}")
    print(f"  With pytestmark: {len(results) - len(missing_pytestmark)}")
    print(f"  Missing pytestmark: {len(missing_pytestmark)}")
    print(f"  With inline markers: {len(has_inline)}")
    print(f"  With both (potential redundancy): {len(has_both)}")

    # Apply fixes if requested
    if args.fix:
        print(f"\n🔧 Fixing {len(missing_pytestmark)} files...")
        fixed_count = 0
        for marker_info in missing_pytestmark:
            if fix_file(marker_info, args.remove_redundant):
                fixed_count += 1
                print(f"  ✓ Fixed {marker_info.file_path}")

        print(f"\n✅ Fixed {fixed_count} files")

        # Re-analyze to verify
        results = [analyze_file(f) for f in test_files]
        missing_pytestmark = [r for r in results if not r.has_pytestmark]

    # Validate format
    format_errors = {}
    for marker_info in results:
        errors = validate_pytestmark_format(marker_info)
        if errors:
            format_errors[marker_info.file_path] = errors

    if format_errors:
        print(f"\n❌ Format validation errors in {len(format_errors)} files:")
        for file_path, errors in format_errors.items():
            print(f"\n  {file_path}:")
            for error in errors:
                print(f"    - {error}")

    # Report files with both pytestmark and inline markers
    if has_both and not args.remove_redundant:
        print(f"\n⚠️  Files with both pytestmark and inline markers ({len(has_both)}):")
        for marker_info in has_both[:10]:  # Show first 10
            print(f"  {marker_info.file_path}")
        if len(has_both) > 10:
            print(f"  ... and {len(has_both) - 10} more")

    # Final validation
    if missing_pytestmark:
        print(f"\n❌ {len(missing_pytestmark)} files still missing pytestmark")
        if args.strict:
            return 1
    else:
        print("\n✅ All test files have pytestmark!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
