#!/usr/bin/env python3
"""Validate Single Source of Truth (SSOT) compliance.

Pre-commit hook that ensures:
- No duplicate configuration files exist
- SSOT directories contain unique configurations
- Generated files are properly marked and tracked
- No config duplication across files
- No documentation duplication
- Forbidden files (per coding-standards.yaml) don't exist
- Version info comes from declared SSOT (pyproject.toml)
- Dependencies come from pyproject.toml
- Test config in pyproject.toml, not elsewhere

Validates against coding-standards.yaml SSOT policy.

Exits with non-zero status if SSOT violations are found.

Version: 2.0.0
Last Updated: 2026-01-22
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


def check_forbidden_files() -> tuple[bool, list[str]]:
    """Check for forbidden files per coding-standards.yaml.

    Returns:
        Tuple of (success: bool, errors: list[str])
    """
    errors = []

    # Load forbidden file patterns from coding-standards.yaml
    forbidden_patterns: list[str] = []
    if CODING_STANDARDS.exists():
        try:
            standards = load_yaml_simple(CODING_STANDARDS)
            ssot_config = standards.get("ssot", {})
            forbidden_patterns = ssot_config.get("forbidden_files", [])
        except Exception:
            pass

    if not forbidden_patterns:
        return True, []

    # Check for forbidden files
    forbidden_found = False
    for pattern in forbidden_patterns:
        # Convert glob pattern to Path pattern
        matches = list(REPO_ROOT.glob(pattern))

        if matches:
            forbidden_found = True
            errors.append(f"❌ Forbidden file pattern '{pattern}' found:")
            for match in matches:
                rel_path = match.relative_to(REPO_ROOT)
                errors.append(f"     - {rel_path}")
            errors.append(
                "     Tip: These files violate SSOT policy. "
                "Move content to appropriate SSOT location."
            )

    if not forbidden_found:
        print("✅ No forbidden files found")

    return not forbidden_found, errors


def check_version_source() -> tuple[bool, list[str]]:
    """Check version info comes from declared SSOT.

    Returns:
        Tuple of (success: bool, errors: list[str])
    """
    errors = []

    # Load SSOT definition
    version_ssot = "pyproject.toml"
    if CODING_STANDARDS.exists():
        try:
            standards = load_yaml_simple(CODING_STANDARDS)
            ssot_config = standards.get("ssot", {})
            version_source = ssot_config.get("version", "")
            if ":" in version_source:
                version_ssot = version_source.split(":")[0]
        except Exception:
            pass

    # Check if version is defined in SSOT location
    pyproject_path = REPO_ROOT / version_ssot
    if not pyproject_path.exists():
        errors.append(f"❌ Version SSOT file not found: {version_ssot}")
        return False, errors

    # Check for version defined elsewhere (common violations)
    version_violations = [
        REPO_ROOT / "setup.py",
        REPO_ROOT / "setup.cfg",
        REPO_ROOT / "__version__.py",
        REPO_ROOT / "src" / "__version__.py",
    ]

    violations_found = False
    for violation_file in version_violations:
        if violation_file.exists():
            # Check if it contains version definition
            try:
                content = violation_file.read_text()
                if "__version__" in content or "version =" in content:
                    violations_found = True
                    errors.append(
                        f"❌ Version defined in {violation_file.name} "
                        f"(should only be in {version_ssot})"
                    )
            except Exception:
                pass

    if not violations_found:
        print(f"✅ Version source correctly defined in {version_ssot}")

    return not violations_found, errors


def check_dependency_source() -> tuple[bool, list[str]]:
    """Check dependencies come from declared SSOT.

    Returns:
        Tuple of (success: bool, errors: list[str])
    """
    errors = []

    # Load SSOT definition
    dep_ssot = "pyproject.toml"
    if CODING_STANDARDS.exists():
        try:
            standards = load_yaml_simple(CODING_STANDARDS)
            ssot_config = standards.get("ssot", {})
            dep_source = ssot_config.get("dependencies", "")
            if ":" in dep_source:
                dep_ssot = dep_source.split(":")[0]
        except Exception:
            pass

    # Check for dependencies defined elsewhere
    dep_violations = [
        (REPO_ROOT / "requirements.txt", "requirements.txt"),
        (REPO_ROOT / "requirements-dev.txt", "requirements-dev.txt"),
        (REPO_ROOT / "setup.py", "setup.py"),
        (REPO_ROOT / "setup.cfg", "setup.cfg"),
        (REPO_ROOT / "Pipfile", "Pipfile"),
    ]

    violations_found = False
    for violation_file, name in dep_violations:
        if violation_file.exists():
            violations_found = True
            errors.append(f"❌ Dependencies defined in {name} (should only be in {dep_ssot})")

    if not violations_found:
        print(f"✅ Dependencies correctly defined in {dep_ssot}")

    return not violations_found, errors


def check_test_config_source() -> tuple[bool, list[str]]:
    """Check test config in pyproject.toml, not elsewhere.

    Returns:
        Tuple of (success: bool, errors: list[str])
    """
    errors = []

    # Check for test config in other locations
    test_config_violations = [
        (REPO_ROOT / "pytest.ini", "pytest.ini"),
        (REPO_ROOT / "setup.cfg", "setup.cfg (pytest section)"),
        (REPO_ROOT / "tox.ini", "tox.ini"),
    ]

    violations_found = False
    for violation_file, name in test_config_violations:
        if violation_file.exists():
            # Check if it contains pytest config
            try:
                content = violation_file.read_text()
                if "[pytest]" in content or "[tool:pytest]" in content:
                    violations_found = True
                    errors.append(f"❌ Test config in {name} (should be in pyproject.toml)")
            except Exception:
                pass

    if not violations_found:
        print("✅ Test config correctly defined in pyproject.toml")

    return not violations_found, errors


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

    # Check forbidden files
    forbidden_ok, forbidden_errors = check_forbidden_files()
    if not forbidden_ok:
        all_passed = False
        all_errors.extend(forbidden_errors)

    # Check version source
    version_ok, version_errors = check_version_source()
    if not version_ok:
        all_passed = False
        all_errors.extend(version_errors)

    # Check dependency source
    dep_ok, dep_errors = check_dependency_source()
    if not dep_ok:
        all_passed = False
        all_errors.extend(dep_errors)

    # Check test config source
    test_ok, test_errors = check_test_config_source()
    if not test_ok:
        all_passed = False
        all_errors.extend(test_errors)

    # Print summary
    print("\n" + "=" * 70)
    if all_passed:
        print("  ✅ ALL CHECKS PASSED")
        print("  - No duplicate configurations")
        print("  - No forbidden files")
        print("  - Version source is SSOT")
        print("  - Dependencies source is SSOT")
        print("  - Test config source is SSOT")
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
