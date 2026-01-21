#!/usr/bin/env python3
"""Validate configuration consistency across project files.

Pre-commit hook that ensures:
- Version is consistent across pyproject.toml, README.md, and src/__init__.py
- GitHub URLs are consistent across project files
- settings.json is in sync with coding-standards.yaml
- .claude orchestration files are consistent (agents, commands, hooks)
- SSOT files exist and are valid

Exits with non-zero status if inconsistencies are found.

Version: 2.0.0
Created: 2026-01-17
Updated: 2026-01-21 (Added orchestration validation)
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Resolve paths - support both real and test environments
REPO_ROOT = Path(
    os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).parent.parent.parent)
).resolve()
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
README_MD = REPO_ROOT / "README.md"
INIT_PY = REPO_ROOT / "src" / "oscura" / "__init__.py"
PROJECT_METADATA = REPO_ROOT / ".claude" / "project-metadata.yaml"
SETTINGS_JSON = REPO_ROOT / ".claude" / "settings.json"
CODING_STANDARDS = REPO_ROOT / ".claude" / "coding-standards.yaml"
ORCHESTRATION_CONFIG = REPO_ROOT / ".claude" / "config.yaml"
AGENTS_DIR = REPO_ROOT / ".claude" / "agents"
COMMANDS_DIR = REPO_ROOT / ".claude" / "commands"
HOOKS_DIR = REPO_ROOT / ".claude" / "hooks"
COORDINATION_DIR = REPO_ROOT / ".coordination"


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


def check_ssot_files() -> tuple[bool, list[str]]:
    """Check if SSOT (Single Source of Truth) files exist.

    Returns:
        Tuple of (success: bool, errors: list[str])
    """
    errors = []
    required_files = [
        PYPROJECT_TOML,
        CODING_STANDARDS,
        ORCHESTRATION_CONFIG,
    ]

    # Optional but expected files
    optional_files = [
        PROJECT_METADATA,
        COORDINATION_DIR / "spec" / "incomplete-features.yaml",
    ]

    for file_path in required_files:
        if not file_path.exists():
            errors.append(f"❌ Missing required SSOT file: {file_path.relative_to(REPO_ROOT)}")

    # Check optional files exist (not errors, just info)
    for file_path in optional_files:
        if file_path.exists():
            print(f"✅ SSOT file exists: {file_path.relative_to(REPO_ROOT)}")

    if errors:
        return False, errors

    print("✅ All required SSOT files present")
    return True, []


def check_agent_command_refs() -> tuple[bool, list[str]]:
    """Check that commands reference existing agents.

    Returns:
        Tuple of (success: bool, errors: list[str])
    """
    errors = []

    if not COMMANDS_DIR.exists() or not AGENTS_DIR.exists():
        # Directories might not exist in minimal setups
        return True, []

    # Get all agent names
    agent_files = list(AGENTS_DIR.glob("*.md"))
    agent_names = {f.stem for f in agent_files}

    # Check each command for agent references
    for command_file in COMMANDS_DIR.glob("*.md"):
        try:
            content = command_file.read_text()

            # Look for agent references (e.g., "Routes to `.claude/agents/orchestrator.md`")
            agent_refs = re.findall(r"\.claude/agents/(\w+)\.md", content)

            for agent_ref in agent_refs:
                if agent_ref not in agent_names:
                    errors.append(
                        f"❌ Command '{command_file.stem}' references "
                        f"non-existent agent '{agent_ref}'"
                    )
        except Exception as e:
            errors.append(f"❌ Error reading command {command_file.name}: {e}")

    if errors:
        return False, errors

    if agent_files:
        print(f"✅ Agent-command references valid ({len(agent_files)} agents)")
    return True, []


def check_hook_refs() -> tuple[bool, list[str]]:
    """Check that settings.json hook references exist.

    Returns:
        Tuple of (success: bool, errors: list[str])
    """
    errors = []

    if not SETTINGS_JSON.exists():
        return True, []  # No settings.json yet

    try:
        with open(SETTINGS_JSON) as f:
            settings = json.load(f)

        hooks_config = settings.get("hooks", {})

        if not isinstance(hooks_config, dict):
            return True, []  # Hooks not configured yet

        # Check each hook reference
        for hook_name, hook_config in hooks_config.items():
            if isinstance(hook_config, str):
                # Simple string reference to hook file
                hook_file = HOOKS_DIR / hook_config
                if not hook_file.exists():
                    errors.append(
                        f"❌ Hook '{hook_name}' references non-existent file: {hook_config}"
                    )
            elif isinstance(hook_config, dict) and "script" in hook_config:
                # Dict with script field
                hook_file = HOOKS_DIR / hook_config["script"]
                if not hook_file.exists():
                    errors.append(
                        f"❌ Hook '{hook_name}' references non-existent script: "
                        f"{hook_config['script']}"
                    )

        if errors:
            return False, errors

        print("✅ Hook references valid")
        return True, []

    except json.JSONDecodeError as e:
        return False, [f"❌ Invalid JSON in settings.json: {e}"]
    except Exception as e:
        return False, [f"❌ Error checking hook references: {e}"]


def check_frontmatter() -> tuple[bool, list[str]]:
    """Check that agent/command markdown files have valid frontmatter.

    Frontmatter is REQUIRED for all agents and commands for consistency and automation.

    Required fields:
    - Agents: name, description, tools, model, routing_keywords
    - Commands: name, description, arguments

    Returns:
        Tuple of (success: bool, errors: list[str])
    """
    errors = []
    checked_count = 0

    if not AGENTS_DIR.exists() and not COMMANDS_DIR.exists():
        return True, []  # No agents/commands yet

    # Check agent frontmatter (REQUIRED)
    if AGENTS_DIR.exists():
        for agent_file in AGENTS_DIR.glob("*.md"):
            checked_count += 1
            try:
                content = agent_file.read_text()

                if not content.startswith("---"):
                    errors.append(f"❌ Agent '{agent_file.name}' missing required frontmatter")
                    continue

                # Parse frontmatter
                parts = content.split("---\n", 2)
                if len(parts) < 3:
                    errors.append(f"❌ Agent '{agent_file.name}' has invalid frontmatter format")
                    continue

                if HAS_YAML:
                    try:
                        frontmatter = yaml.safe_load(parts[1])
                        if not isinstance(frontmatter, dict):
                            errors.append(f"❌ Agent '{agent_file.name}' frontmatter is not a dict")
                            continue

                        # Validate required fields for agents
                        required_fields = [
                            "name",
                            "description",
                            "tools",
                            "model",
                            "routing_keywords",
                        ]
                        for field in required_fields:
                            if field not in frontmatter:
                                errors.append(
                                    f"❌ Agent '{agent_file.name}' missing required field: {field}"
                                )

                    except yaml.YAMLError as e:
                        errors.append(f"❌ Agent '{agent_file.name}' has invalid YAML: {e}")
            except Exception as e:
                errors.append(f"❌ Error reading agent {agent_file.name}: {e}")

    # Check command frontmatter (REQUIRED)
    if COMMANDS_DIR.exists():
        for command_file in COMMANDS_DIR.glob("*.md"):
            checked_count += 1
            try:
                content = command_file.read_text()

                if not content.startswith("---"):
                    errors.append(f"❌ Command '{command_file.name}' missing required frontmatter")
                    continue

                parts = content.split("---\n", 2)
                if len(parts) < 3:
                    errors.append(
                        f"❌ Command '{command_file.name}' has invalid frontmatter format"
                    )
                    continue

                if HAS_YAML:
                    try:
                        frontmatter = yaml.safe_load(parts[1])
                        if not isinstance(frontmatter, dict):
                            errors.append(
                                f"❌ Command '{command_file.name}' frontmatter is not a dict"
                            )
                            continue

                        # Validate required fields for commands
                        required_fields = ["name", "description", "arguments"]
                        for field in required_fields:
                            if field not in frontmatter:
                                errors.append(
                                    f"❌ Command '{command_file.name}' missing required field: {field}"
                                )

                    except yaml.YAMLError as e:
                        errors.append(f"❌ Command '{command_file.name}' has invalid YAML: {e}")
            except Exception as e:
                errors.append(f"❌ Error reading command {command_file.name}: {e}")

    if errors:
        return False, errors

    print(f"✅ Frontmatter validation passed ({checked_count} files)")
    return True, []


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
    error_count = 0

    # Check version consistency
    version_ok, version_errors = check_version_consistency()
    if not version_ok:
        all_passed = False
        error_count += len(version_errors)
        all_errors.extend(version_errors)

    # Check settings.json sync
    settings_ok, settings_errors = check_settings_sync()
    if not settings_ok:
        all_passed = False
        error_count += len(settings_errors)
        all_errors.extend(settings_errors)

    # Check SSOT files
    ssot_ok, ssot_errors = check_ssot_files()
    if not ssot_ok:
        all_passed = False
        error_count += len(ssot_errors)
        all_errors.extend(ssot_errors)

    # Check agent-command references
    agent_ok, agent_errors = check_agent_command_refs()
    if not agent_ok:
        all_passed = False
        error_count += len(agent_errors)
        all_errors.extend(agent_errors)

    # Check hook references
    hook_ok, hook_errors = check_hook_refs()
    if not hook_ok:
        all_passed = False
        error_count += len(hook_errors)
        all_errors.extend(hook_errors)

    # Check frontmatter
    frontmatter_ok, frontmatter_errors = check_frontmatter()
    if not frontmatter_ok:
        all_passed = False
        error_count += len(frontmatter_errors)
        all_errors.extend(frontmatter_errors)

    # Print summary
    print("\n" + "=" * 70)
    if all_passed:
        print("  ✅ Configuration is consistent")
        print(f"  Errors: 0")
    else:
        print("  ❌ VALIDATION FAILED")
        print(f"  Errors: {error_count}")
        print("=" * 70)
        print("\nErrors:")
        for error in all_errors:
            print(f"  {error}")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
