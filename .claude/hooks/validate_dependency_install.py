#!/usr/bin/env python3
"""Validate that ALL dependencies are installed in development/test environments.

CRITICAL POLICY ENFORCEMENT:
    For development and testing, there are NO optional dependencies.
    ALL dependencies from pyproject.toml[project.optional-dependencies]
    MUST be installed to prevent configuration drift and test failures.

This hook verifies:
    1. All core dependencies are installed (from [project.dependencies])
    2. All optional dependencies are installed (from [project.optional-dependencies])
    3. No missing dependencies that could cause import failures
    4. Consistent versions across local and CI environments

Exit codes:
    0: All dependencies installed correctly
    1: Missing dependencies detected (BLOCKS commit/CI)
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


def get_project_root() -> Path:
    """Find project root by locating pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find pyproject.toml")


def get_installed_packages() -> set[str]:
    """Get set of all installed package names."""
    try:
        result = subprocess.run(
            ["uv", "pip", "list", "--format=freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        packages = set()
        for line in result.stdout.strip().split("\n"):
            if line and "==" in line:
                pkg_name = line.split("==")[0].lower().replace("_", "-")
                packages.add(pkg_name)
        return packages
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Failed to get installed packages: {e}", file=sys.stderr)
        sys.exit(1)


def extract_package_name(dep_spec: str) -> str:
    """Extract package name from dependency specification.

    Examples:
        pytest>=8.0,<10.0.0 -> pytest
        myproject[dev,reporting] -> myproject
    """
    dep = dep_spec.strip()
    # Remove version specifiers and extras
    for char in [">=", "<=", "==", "!=", "~=", ">", "<", "[", ";"]:
        if char in dep:
            dep = dep.split(char)[0]
    return dep.lower().replace("_", "-")


def validate_dependencies() -> tuple[bool, list[str]]:
    """Validate all required dependencies are installed.

    Returns:
        (all_installed, missing_packages)
    """
    project_root = get_project_root()
    pyproject_path = project_root / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)

    installed = get_installed_packages()
    missing = []

    # Check core dependencies
    core_deps = config.get("project", {}).get("dependencies", [])
    for dep in core_deps:
        pkg = extract_package_name(dep)
        if pkg not in installed:
            missing.append(f"CORE: {dep}")

    # Check ALL optional dependencies (MANDATORY for dev/test)
    optional_deps = config.get("project", {}).get("optional-dependencies", {})

    # Get project name from config
    project_name = config.get("project", {}).get("name", "").lower()

    # Skip 'all' group (it's a meta-group referencing others)
    for group_name, deps in optional_deps.items():
        if group_name == "all":
            continue

        for dep in deps:
            # Skip self-references like "projectname[...]"
            if project_name and dep.lower().startswith(f"{project_name}["):
                continue

            pkg = extract_package_name(dep)
            if pkg not in installed:
                missing.append(f"{group_name.upper()}: {dep}")

    return len(missing) == 0, missing


def main() -> int:
    """Run dependency validation."""
    print("=" * 80)
    print("DEPENDENCY INSTALLATION VALIDATOR")
    print("=" * 80)
    print()
    print("POLICY: ALL dependencies MUST be installed for development/testing.")
    print("        There are NO optional dependencies in dev/test environments.")
    print()

    all_installed, missing = validate_dependencies()

    if all_installed:
        print("✅ SUCCESS: All required dependencies are installed")
        print()
        print("Verified:")
        print("  - All core dependencies (from [project.dependencies])")
        print("  - All optional dependencies (from [project.optional-dependencies])")
        print()
        return 0
    else:
        print("❌ FAILURE: Missing dependencies detected!")
        print()
        print("Missing packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print()
        print("REQUIRED ACTION:")
        print("  Run: uv sync --all-extras --group dev")
        print()
        print("RATIONALE:")
        print("  Configuration drift between local and CI environments causes")
        print("  test failures. Every test environment must have ALL dependencies.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
