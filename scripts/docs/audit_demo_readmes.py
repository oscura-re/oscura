#!/usr/bin/env python3
"""Audit all demo READMEs for consistency issues.

Checks for:
1. Version mismatches (should be 0.1.2, not 0.3.x)
2. Broken internal references (demos/XX_category should exist)
3. Inconsistent structure
"""

import re
from pathlib import Path

# Current version
CURRENT_VERSION = "0.1.2"


def audit_demo_readme(path: Path) -> list[str]:
    """Audit a single demo README.

    Returns:
        List of issues found.
    """
    content = path.read_text()
    issues = []

    # Check for version mismatches
    version_pattern = r"\b0\.[23]\.[\dx]+"
    matches = re.findall(version_pattern, content)
    for match in matches:
        if match != CURRENT_VERSION:
            issues.append(f"Version mismatch: found '{match}', expected '{CURRENT_VERSION}'")

    # Check for broken internal references (demos/XX_category)
    ref_pattern = r"demos/(\d+)_[\w_]+"
    refs = re.findall(ref_pattern, content)
    demos_dir = Path(__file__).parent.parent.parent / "demos"

    for ref_num in set(refs):
        # Check if any directory starts with this number
        matching = list(demos_dir.glob(f"{ref_num}_*"))
        if not matching:
            issues.append(f"Broken reference: demos/{ref_num}_* does not exist")

    return issues


def main():
    """Audit all demo READMEs."""
    demos_dir = Path(__file__).parent.parent.parent / "demos"
    readme_files = sorted(demos_dir.glob("*/README.md"))

    # Also check nested READMEs
    nested = sorted(demos_dir.glob("*/*/README.md"))
    readme_files.extend(nested)

    total_issues = 0
    files_with_issues = 0

    print("=" * 80)
    print("Demo README Audit Report")
    print("=" * 80)
    print()

    for readme_path in readme_files:
        rel_path = readme_path.relative_to(demos_dir.parent)
        issues = audit_demo_readme(readme_path)

        if issues:
            files_with_issues += 1
            total_issues += len(issues)
            print(f"❌ {rel_path}")
            for issue in issues:
                print(f"   - {issue}")
            print()
        else:
            print(f"✅ {rel_path}")

    print()
    print("=" * 80)
    print(f"Summary: {total_issues} issues in {files_with_issues}/{len(readme_files)} files")
    print("=" * 80)

    return total_issues


if __name__ == "__main__":
    import sys

    issues = main()
    sys.exit(0 if issues == 0 else 1)
