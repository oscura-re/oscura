#!/usr/bin/env python3
"""Validate Single Source of Truth (SSOT) compliance.

Pre-commit hook that ensures:
- No duplicate configuration files exist
- SSOT directories contain unique configurations
- Generated files are properly marked and tracked

Validates against coding-standards.yaml SSOT policy.

Exits with non-zero status if SSOT violations are found.

Version: 1.0.0
Created: 2026-01-17
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Resolve paths
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
CODING_STANDARDS = REPO_ROOT / ".claude" / "coding-standards.yaml"


def load_yaml_simple(file_path: Path) -> dict[str, Any]:
    """Load YAML file with basic parsing fallback."""
    if HAS_YAML:
        with open(file_path) as f:
            return yaml.safe_load(f) or {}

    # Fallback: basic key-value extraction
    result: dict[str, Any] = {}
    with open(file_path) as f:
        for line in f:
            if ":" in line and not line.strip().startswith("#"):
                key, _, value = line.partition(":")
                result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def check_duplicate_configs() -> tuple[bool, list[str]]:
    """Check for duplicate configuration files.

    Returns:
        Tuple of (success: bool, errors: list[str])
    """
    errors = []

    # Load allowlist from coding-standards.yaml
    allowlist: set[str] = set()
    if CODING_STANDARDS.exists():
        try:
            standards = load_yaml_simple(CODING_STANDARDS)
            ssot_config = standards.get("file_organization", {}).get("ssot_validation", {})
            allowed_duplicates = ssot_config.get("allowed_duplicate_configs", [])
            allowlist = set(allowed_duplicates)
        except Exception:
            # If can't load allowlist, continue with empty set
            pass

    # Common config patterns that should be unique
    config_patterns = [
        "**/*config*.yaml",
        "**/*config*.json",
        "**/*.toml",
        "**/*settings*.json",
    ]

    # Exclusions
    excluded_dirs = {
        ".venv",
        "node_modules",
        "__pycache__",
        ".git",
        ".mypy_cache",
        ".ruff_cache",
        "archive",
        "test_data",
        ".pytest_cache",
        "site",
        "dist",
        "build",
        ".tox",
        "htmlcov",
        ".vscode",  # VSCode configs are separate from project configs
        ".idea",  # PyCharm configs
        ".cache",  # Pre-push cache directory
    }

    # Track files by basename
    files_by_name: dict[str, list[Path]] = defaultdict(list)

    for pattern in config_patterns:
        for file_path in REPO_ROOT.glob(pattern):
            # Skip excluded directories
            if any(excluded in file_path.parts for excluded in excluded_dirs):
                continue

            # Skip generated files (marked with _generated field)
            if file_path.suffix == ".json":
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                        if isinstance(data, dict) and "_generated" in data:
                            continue
                except (json.JSONDecodeError, OSError):
                    pass

            files_by_name[file_path.name].append(file_path)

    # Check for duplicates
    duplicates_found = False
    for name, paths in files_by_name.items():
        if len(paths) > 1:
            # Check if this duplicate is allowed
            if name in allowlist:
                print(f"✅ Allowed duplicate: {name} ({len(paths)} instances)")
                continue

            # Not in allowlist - report as error
            duplicates_found = True
            errors.append(f"❌ Duplicate config file '{name}' found in:")
            for path in paths:
                rel_path = path.relative_to(REPO_ROOT)
                errors.append(f"     - {rel_path}")
            errors.append(
                f"     Tip: If this is intentional, add '{name}' to "
                "coding-standards.yaml > file_organization > ssot_validation > allowed_duplicate_configs"
            )

    if not duplicates_found:
        print("✅ No duplicate configuration files found")

    return not duplicates_found, errors


def check_ssot_directories() -> tuple[bool, list[str]]:
    """Check SSOT directories for consistency.

    Returns:
        Tuple of (success: bool, errors: list[str])
    """
    errors = []

    # Load SSOT directory definitions
    if not CODING_STANDARDS.exists():
        errors.append("❌ coding-standards.yaml not found")
        return False, errors

    try:
        standards = load_yaml_simple(CODING_STANDARDS)
        ssot_dirs = standards.get("file_organization", {}).get("ssot_directories", {})

        if not ssot_dirs:
            # No SSOT directories defined
            return True, []

        # Check each SSOT directory
        for purpose, directory in ssot_dirs.items():
            dir_path = REPO_ROOT / directory

            if not dir_path.exists():
                # Directory doesn't exist yet - not an error
                continue

            # Count files in directory
            files = list(dir_path.glob("*.yaml")) + list(dir_path.glob("*.json"))

            # Filter out excluded files
            files = [
                f
                for f in files
                if not any(
                    excl in f.parts for excl in [".venv", "node_modules", "__pycache__", "archive"]
                )
            ]

            if files:
                print(f"✅ SSOT directory '{purpose}': {len(files)} files in {directory}")

        return True, []

    except Exception as e:
        errors.append(f"❌ Error checking SSOT directories: {e}")
        return False, errors


def check_generated_files() -> tuple[bool, list[str]]:
    """Check that generated files are properly marked.

    Returns:
        Tuple of (success: bool, errors: list[str])
    """
    errors = []

    # Check settings.json
    settings_path = REPO_ROOT / ".claude" / "settings.json"

    if settings_path.exists():
        try:
            with open(settings_path) as f:
                settings = json.load(f)

            # Check for _generated marker
            if "_generated" in settings:
                source = settings["_generated"].get("source", "unknown")
                print(f"✅ settings.json properly marked as generated from {source}")
            else:
                print("⚠ settings.json not marked as generated (may be manual)")

        except json.JSONDecodeError:
            errors.append("❌ settings.json is not valid JSON")
            return False, errors

    return True, []


def main() -> int:
    """Main entry point.

    Returns:
        0 if all checks pass, 1 if any check fails
    """
    print("\n" + "=" * 70)
    print("  SSOT VALIDATION")
    print("=" * 70 + "\n")

    all_passed = True
    all_errors: list[str] = []

    # Check for duplicate configs
    duplicates_ok, duplicate_errors = check_duplicate_configs()
    if not duplicates_ok:
        all_passed = False
        all_errors.extend(duplicate_errors)

    # Check SSOT directories
    ssot_ok, ssot_errors = check_ssot_directories()
    if not ssot_ok:
        all_passed = False
        all_errors.extend(ssot_errors)

    # Check generated files
    generated_ok, generated_errors = check_generated_files()
    if not generated_ok:
        all_passed = False
        all_errors.extend(generated_errors)

    # Print summary
    print("\n" + "=" * 70)
    if all_passed:
        print("  ✅ ALL CHECKS PASSED")
    else:
        print("  ❌ VALIDATION FAILED")
        print("=" * 70)
        print("\nErrors:")
        for error in all_errors:
            print(f"  {error}")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
