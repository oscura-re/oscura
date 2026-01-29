#!/usr/bin/env python3
"""Identify source files without corresponding tests.

This script analyzes src/oscura/ and tests/ to find gaps in test coverage.
"""

from pathlib import Path
from typing import NamedTuple


class CoverageGap(NamedTuple):
    """Represents a source file without test coverage."""

    source_file: Path
    expected_test_paths: list[Path]
    category: str  # "unit", "integration", or "both"


def find_untested_files() -> list[CoverageGap]:
    """Find all source files without corresponding tests.

    Returns:
        List of CoverageGap objects for files missing tests

    Example:
        >>> gaps = find_untested_files()
        >>> for gap in gaps[:5]:
        ...     print(f"{gap.source_file} -> {gap.expected_test_paths}")
    """
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src" / "oscura"
    tests_dir = project_root / "tests"

    # Find all Python source files
    source_files = list(src_dir.rglob("*.py"))

    gaps = []
    for source_file in source_files:
        # Calculate relative path from src/oscura/
        rel_path = source_file.relative_to(src_dir)

        # Skip __init__.py files (often empty or minimal)
        if source_file.name == "__init__.py":
            continue

        # Expected test file paths
        unit_test_path = tests_dir / "unit" / rel_path.parent / f"test_{rel_path.name}"
        integration_test_path = (
            tests_dir / "integration" / rel_path.parent / f"test_{rel_path.name}"
        )

        # Check if tests exist
        has_unit = unit_test_path.exists()
        has_integration = integration_test_path.exists()

        if not has_unit and not has_integration:
            # Completely untested
            gaps.append(
                CoverageGap(
                    source_file=source_file,
                    expected_test_paths=[unit_test_path, integration_test_path],
                    category="both",
                )
            )
        elif not has_unit:
            # Missing unit tests
            gaps.append(
                CoverageGap(
                    source_file=source_file,
                    expected_test_paths=[unit_test_path],
                    category="unit",
                )
            )

    return gaps


def main() -> None:
    """Generate report of untested files."""
    gaps = find_untested_files()

    # Categorize by type
    completely_untested = [g for g in gaps if g.category == "both"]
    missing_unit = [g for g in gaps if g.category == "unit"]

    print("Coverage Gap Analysis")
    print("=" * 80)
    print(f"Total untested files: {len(gaps)}")
    print(f"  - Completely untested: {len(completely_untested)}")
    print(f"  - Missing unit tests only: {len(missing_unit)}")
    print()

    # Group by directory
    by_directory: dict[str, list[CoverageGap]] = {}
    for gap in completely_untested:
        parent = gap.source_file.parent.relative_to(Path(__file__).parent.parent / "src" / "oscura")
        key = str(parent) if str(parent) != "." else "(root)"
        by_directory.setdefault(key, []).append(gap)

    # Sort by directory with most gaps first
    sorted_dirs = sorted(by_directory.items(), key=lambda x: len(x[1]), reverse=True)

    print("Untested Files by Directory:")
    print("-" * 80)
    for directory, dir_gaps in sorted_dirs:
        print(f"\n{directory}/ ({len(dir_gaps)} files)")
        for gap in sorted(dir_gaps, key=lambda g: g.source_file.name):
            print(f"  - {gap.source_file.name}")

    # Output actionable list
    print("\n" + "=" * 80)
    print("Actionable Test Creation List")
    print("=" * 80)
    for gap in sorted(completely_untested, key=lambda g: str(g.source_file)):
        rel_path = gap.source_file.relative_to(Path(__file__).parent.parent / "src" / "oscura")
        test_path = gap.expected_test_paths[0].relative_to(Path(__file__).parent.parent)
        print(f"{rel_path} -> {test_path}")


if __name__ == "__main__":
    main()
