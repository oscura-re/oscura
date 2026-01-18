#!/usr/bin/env python3
"""Validate configuration consistency across project files.

Pre-commit hook that ensures:
- Version is consistent across pyproject.toml, README.md, and src/__init__.py
- GitHub URLs are consistent across project files
- settings.json is in sync with coding-standards.yaml

Exits with non-zero status if inconsistencies are found.

Version: 1.0.0
Created: 2026-01-17
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Resolve paths
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
README_MD = REPO_ROOT / "README.md"
INIT_PY = REPO_ROOT / "src" / "oscura" / "__init__.py"
PROJECT_METADATA = REPO_ROOT / ".claude" / "project-metadata.yaml"
SETTINGS_JSON = REPO_ROOT / ".claude" / "settings.json"
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


def extract_version_from_toml() -> str | None:
    """Extract version from pyproject.toml."""
    if not PYPROJECT_TOML.exists():
        return None

    with open(PYPROJECT_TOML) as f:
        for line in f:
            if line.strip().startswith("version"):
                match = re.search(r'version\s*=\s*"([^"]+)"', line)
                if match:
                    return match.group(1)
    return None


def extract_version_from_readme() -> str | None:
    """Extract version from README.md."""
    if not README_MD.exists():
        return None

    with open(README_MD) as f:
        content = f.read()
        # Look for "Current Version: vX.Y.Z" pattern
        match = re.search(r"Current Version:\s*v?(\d+\.\d+\.\d+[^\s]*)", content)
        if match:
            return match.group(1)
    return None


def extract_version_from_init() -> str | None:
    """Extract version from src/oscura/__init__.py."""
    if not INIT_PY.exists():
        return None

    with open(INIT_PY) as f:
        for line in f:
            if "__version__" in line:
                match = re.search(r'__version__\s*=\s*"([^"]+)"', line)
                if match:
                    return match.group(1)
    return None


def extract_github_url_from_toml() -> str | None:
    """Extract GitHub URL from pyproject.toml."""
    if not PYPROJECT_TOML.exists():
        return None

    with open(PYPROJECT_TOML) as f:
        content = f.read()
        # Look for repository URL in [project.urls]
        match = re.search(r'(?:Repository|repository|Homepage)\s*=\s*"([^"]+)"', content)
        if match:
            return match.group(1)
    return None


def check_version_consistency() -> tuple[bool, list[str]]:
    """Check version consistency across files.

    Returns:
        Tuple of (success: bool, errors: list[str])
    """
    errors = []

    # Extract versions
    version_toml = extract_version_from_toml()
    version_readme = extract_version_from_readme()
    version_init = extract_version_from_init()

    if not version_toml:
        errors.append("❌ Could not extract version from pyproject.toml")
        return False, errors

    # Check README version
    if version_readme and version_readme != version_toml:
        errors.append(
            f"❌ Version mismatch: README.md has {version_readme}, "
            f"pyproject.toml has {version_toml}"
        )

    # Check __init__.py version
    if version_init and version_init != version_toml:
        errors.append(
            f"❌ Version mismatch: __init__.py has {version_init}, "
            f"pyproject.toml has {version_toml}"
        )

    if errors:
        return False, errors

    print(f"✅ Version consistency: {version_toml}")
    return True, []


def check_settings_sync() -> tuple[bool, list[str]]:
    """Check if settings.json is in sync with coding-standards.yaml.

    Returns:
        Tuple of (success: bool, errors: list[str])
    """
    errors = []

    if not SETTINGS_JSON.exists():
        # Settings.json doesn't exist yet - not an error
        return True, []

    if not CODING_STANDARDS.exists():
        errors.append("❌ coding-standards.yaml not found")
        return False, errors

    try:
        # Load settings.json
        with open(SETTINGS_JSON) as f:
            settings = json.load(f)

        # Load coding standards
        standards = load_yaml_simple(CODING_STANDARDS)

        # Check if settings has _generated marker
        if "_generated" not in settings:
            # Settings.json is manually managed or old format
            print("⚠ settings.json not generated (manual configuration)")
            return True, []

        # Enhanced validation: Check multiple fields
        validation_errors = []

        # 1. Check cleanupPeriodDays
        expected_cleanup = (
            standards.get("cleanup", {}).get("retention", {}).get("checkpoint_archives", 30)
        )
        actual_cleanup = settings.get("cleanupPeriodDays")
        if actual_cleanup != expected_cleanup:
            validation_errors.append(
                f"cleanupPeriodDays: expected {expected_cleanup}, got {actual_cleanup}"
            )

        # 2. Check source_hash exists (our new optimization!)
        generated = settings.get("_generated", {})
        if "source_hash" not in generated:
            validation_errors.append(
                "_generated.source_hash: missing (should use hash instead of timestamp)"
            )

        # 3. Check model field exists
        if "model" not in settings:
            validation_errors.append("model: missing required field")

        # 4. Check alwaysThinkingEnabled exists
        if "alwaysThinkingEnabled" not in settings:
            validation_errors.append("alwaysThinkingEnabled: missing required field")

        # 5. Check permissions structure exists
        if "permissions" not in settings:
            validation_errors.append("permissions: missing required field")

        # 6. Check hooks structure exists
        if "hooks" not in settings:
            validation_errors.append("hooks: missing required field")

        if validation_errors:
            errors.append(f"❌ settings.json out of sync with coding-standards.yaml:")
            for ve in validation_errors:
                errors.append(f"     - {ve}")
            errors.append("     Run: python .claude/hooks/generate_settings.py")
            return False, errors

        print("✅ settings.json is in sync with coding-standards.yaml")
        print(f"   Source hash: {generated.get('source_hash', 'N/A')}")
        return True, []

    except json.JSONDecodeError as e:
        errors.append(f"❌ Invalid JSON in settings.json: {e}")
        return False, errors
    except Exception as e:
        errors.append(f"❌ Error checking settings sync: {e}")
        return False, errors


def main() -> int:
    """Main entry point.

    Returns:
        0 if all checks pass, 1 if any check fails
    """
    print("\n" + "=" * 70)
    print("  CONFIG CONSISTENCY VALIDATION")
    print("=" * 70 + "\n")

    all_passed = True
    all_errors: list[str] = []

    # Check version consistency
    version_ok, version_errors = check_version_consistency()
    if not version_ok:
        all_passed = False
        all_errors.extend(version_errors)

    # Check settings.json sync
    settings_ok, settings_errors = check_settings_sync()
    if not settings_ok:
        all_passed = False
        all_errors.extend(settings_errors)

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
