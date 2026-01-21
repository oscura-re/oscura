#!/usr/bin/env python3
"""Synchronize version numbers across all project files.

This hook ensures version consistency between pyproject.toml (SSOT)
and all other files that reference the version:
- src/oscura/__init__.py
- src/oscura/automotive/__init__.py
- docs/**/*.md files

Exit codes:
- 0: All versions synchronized successfully
- 1: Version mismatch found and fixed (staged files modified)
- 2: Configuration error or critical failure
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for Python 3.10
    except ImportError:
        import toml as tomllib  # Legacy fallback


def get_version_from_pyproject() -> str:
    """Get the version from pyproject.toml (SSOT)."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("ERROR: pyproject.toml not found", file=sys.stderr)
        sys.exit(2)

    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        version = data["project"]["version"]
        return version
    except (KeyError, Exception) as e:
        print(f"ERROR: Could not read version from pyproject.toml: {e}", file=sys.stderr)
        sys.exit(2)


def update_python_init_version(file_path: Path, target_version: str) -> bool:
    """Update __version__ in a Python __init__.py file."""
    if not file_path.exists():
        return False

    content = file_path.read_text()
    pattern = r'__version__\s*=\s*["\']([^"\']+)["\']'

    match = re.search(pattern, content)
    if not match:
        return False

    current_version = match.group(1)
    if current_version == target_version:
        return False

    # Version mismatch - update it
    new_content = re.sub(pattern, f'__version__ = "{target_version}"', content)
    file_path.write_text(new_content)
    print(f"Updated {file_path}: {current_version} → {target_version}")
    return True


def update_doc_version(file_path: Path, target_version: str) -> bool:
    """Update version references in documentation files."""
    if not file_path.exists():
        return False

    content = file_path.read_text()
    modified = False

    # Pattern: **Version**: X.Y.Z
    version_pattern = r"\*\*Version\*\*:\s*(\d+\.\d+\.\d+)"
    matches = list(re.finditer(version_pattern, content))
    if matches:
        for match in matches:
            current_version = match.group(1)
            if current_version != target_version:
                content = content.replace(
                    f"**Version**: {current_version}", f"**Version**: {target_version}"
                )
                modified = True

    # Pattern: Oscura X.Y.Z
    oscura_pattern = r"Oscura\s+(\d+\.\d+\.\d+)"
    matches = list(re.finditer(oscura_pattern, content))
    if matches:
        for match in matches:
            current_version = match.group(1)
            if current_version != target_version:
                content = content.replace(f"Oscura {current_version}", f"Oscura {target_version}")
                modified = True

    if modified:
        file_path.write_text(content)
        print(f"Updated {file_path}: documentation version → {target_version}")

    return modified


def sync_versions() -> int:
    """Sync all version references to match pyproject.toml.

    Returns:
        0 if all versions match
        1 if versions were updated
        2 if error occurred
    """
    target_version = get_version_from_pyproject()
    print(f"Target version (from pyproject.toml): {target_version}")

    modified_files = []

    # Update Python __init__.py files
    python_files = [
        Path("src/oscura/__init__.py"),
        Path("src/oscura/automotive/__init__.py"),
    ]

    for file_path in python_files:
        if update_python_init_version(file_path, target_version):
            modified_files.append(file_path)

    # Update documentation files
    docs_path = Path("docs")
    if docs_path.exists():
        for doc_file in docs_path.rglob("*.md"):
            if update_doc_version(doc_file, target_version):
                modified_files.append(doc_file)

    if modified_files:
        print(f"\n✗ Version mismatch found and fixed in {len(modified_files)} file(s)")
        print("Modified files have been updated. Please stage them and commit again.")
        return 1

    print("\n✓ All version references are synchronized")
    return 0


if __name__ == "__main__":
    sys.exit(sync_versions())
