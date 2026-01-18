#!/usr/bin/env python3
"""
Generate settings.json from coding-standards.yaml (SSOT)

Ensures that .claude/settings.json is always derived from the authoritative
coding-standards.yaml configuration, preventing configuration drift.

Version: 1.0.0
Created: 2025-12-25
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# =============================================================================
# Configuration
# =============================================================================

# Resolve paths from script location
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR", str(REPO_ROOT)))

CODING_STANDARDS_FILE = PROJECT_DIR / ".claude" / "coding-standards.yaml"
SETTINGS_FILE = PROJECT_DIR / ".claude" / "settings.json"
BACKUP_FILE = PROJECT_DIR / ".claude" / "settings.json.bak"
LOG_FILE = PROJECT_DIR / ".claude" / "hooks" / "hook.log"
HASH_FILE = PROJECT_DIR / ".claude" / "hooks" / ".settings_hash"

# =============================================================================
# Logging Setup
# =============================================================================

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler()],
)
logger = logging.getLogger("generate_settings")


# =============================================================================
# YAML Loading (with fallback)
# =============================================================================


def load_yaml(file_path: Path) -> dict[str, Any]:
    """Load YAML file with PyYAML or fallback to basic parsing."""
    try:
        import yaml

        with open(file_path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("PyYAML not installed. Using basic YAML parsing (limited support).")
        return _basic_yaml_parse(file_path)


def _basic_yaml_parse(file_path: Path) -> dict[str, Any]:
    """Very basic YAML parser for simple key-value structures.

    This is a fallback when PyYAML is not available. Only handles simple cases.
    """
    result: dict[str, Any] = {}
    current_section = None
    current_list: list[Any] | None = None

    with open(file_path) as f:
        for line in f:
            line = line.rstrip()

            # Skip empty lines and comments
            if not line or line.strip().startswith("#"):
                continue

            # Count indentation
            indent = len(line) - len(line.lstrip())
            stripped = line.strip()

            # Top-level key
            if indent == 0 and ":" in stripped:
                key, _, value = stripped.partition(":")
                key = key.strip()
                value = value.strip()

                if value:
                    result[key] = value.strip('"').strip("'")
                else:
                    result[key] = {}
                    current_section = key
                    current_list = None

            # Section content (simplified - just store as string)
            elif current_section is not None:
                if stripped.startswith("-"):
                    if current_list is None:
                        current_list = []
                        result[current_section] = current_list
                    item = stripped[1:].strip().strip('"').strip("'")
                    if item:
                        current_list.append(item)
                elif ":" in stripped:
                    key, _, value = stripped.partition(":")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if isinstance(result[current_section], dict):
                        result[current_section][key] = value

    return result


# =============================================================================
# Settings Generation
# =============================================================================


def get_current_settings() -> dict[str, Any]:
    """Load current settings.json if it exists."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                settings: dict[str, Any] = json.load(f)
                return settings
        except json.JSONDecodeError:
            logger.warning("Current settings.json is corrupted")
            return {}
    return {}


def compute_hash(data: dict[str, Any]) -> str:
    """Compute hash of dictionary for change detection."""
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def atomic_write(file_path: Path, content: str) -> None:
    """Write file atomically with backup."""
    # Create backup of existing file
    if file_path.exists():
        shutil.copy2(file_path, BACKUP_FILE)
        logger.info(f"Backed up existing file to {BACKUP_FILE}")

    # Write to temp file first
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=file_path.parent, delete=False
    ) as f:
        f.write(content)
        temp_path = Path(f.name)

    # Atomic replace
    try:
        temp_path.replace(file_path)
    except OSError as e:
        # Restore from backup on failure
        if BACKUP_FILE.exists():
            shutil.copy2(BACKUP_FILE, file_path)
            logger.error(f"Write failed, restored from backup: {e}")
        temp_path.unlink(missing_ok=True)
        raise


def generate_settings(dry_run: bool = False, output: str | None = None) -> dict[str, Any]:
    """Generate settings.json from coding-standards.yaml.

    Args:
        dry_run: If True, don't write files, just report what would change
        output: Optional alternative output path

    Returns:
        Generated settings dictionary
    """
    # Load coding standards
    if not CODING_STANDARDS_FILE.exists():
        logger.error(f"Coding standards file not found: {CODING_STANDARDS_FILE}")
        return {"error": "coding-standards.yaml not found"}

    standards = load_yaml(CODING_STANDARDS_FILE)

    # Get current settings to preserve non-generated fields
    current = get_current_settings()

    # Start with current settings to preserve model, permissions, etc.
    settings: dict[str, Any] = current.copy()

    # Update with generated values
    settings.update(
        {
            # Preserve existing values or use defaults
            "model": current.get("model", "sonnet"),
            "cleanupPeriodDays": standards.get("cleanup", {})
            .get("retention", {})
            .get("checkpoint_archives", 30),
            "alwaysThinkingEnabled": current.get("alwaysThinkingEnabled", True),
            # Mark as generated (using source file hash as version to avoid timestamp churn)
            "_generated": {
                "source": "coding-standards.yaml",
                "source_hash": hashlib.sha256(CODING_STANDARDS_FILE.read_bytes()).hexdigest()[:12],
                "version": "1.0.0",
            },
        }
    )

    # Note: We preserve the existing hooks section from settings.json
    # because it has the actual Claude Code hook format which differs
    # from the simplified format in coding-standards.yaml
    # The coding-standards.yaml documents WHAT should run, while
    # settings.json has the HOW (command format, timeouts, etc.)

    if "hooks" not in settings:
        settings["hooks"] = {}

    # Compute hash to detect changes
    new_hash = compute_hash(settings)
    old_hash = ""
    if HASH_FILE.exists():
        old_hash = HASH_FILE.read_text().strip()

    # Determine output path
    output_path = Path(output) if output else SETTINGS_FILE

    if dry_run:
        logger.info("DRY RUN - Would generate settings:")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Changed: {new_hash != old_hash}")
        return settings

    # Only write if changed
    if new_hash == old_hash and output_path == SETTINGS_FILE:
        logger.info("Settings unchanged, skipping write")
        return settings

    # Write settings
    content = json.dumps(settings, indent=2) + "\n"
    atomic_write(output_path, content)

    # Save hash for change detection
    HASH_FILE.write_text(new_hash)

    logger.info(f"Generated {output_path}")
    return settings


def validate_sync() -> bool:
    """Check if settings.json is in sync with coding-standards.yaml.

    Returns True if in sync, False otherwise.
    """
    if not SETTINGS_FILE.exists():
        logger.warning("settings.json does not exist")
        return False

    if not CODING_STANDARDS_FILE.exists():
        logger.warning("coding-standards.yaml does not exist")
        return False

    # Generate what settings should be
    expected = generate_settings(dry_run=True)
    if "error" in expected:
        return False

    # Load current settings
    current = get_current_settings()

    # Check key fields that should be synced
    sync_fields = ["cleanupPeriodDays"]
    for field in sync_fields:
        if current.get(field) != expected.get(field):
            logger.warning(
                f"Field '{field}' out of sync: current={current.get(field)}, expected={expected.get(field)}"
            )
            return False

    return True


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate settings.json from coding-standards.yaml"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be generated without writing"
    )
    parser.add_argument("--output", "-o", help="Alternative output path")
    parser.add_argument(
        "--validate", "-v", action="store_true", help="Validate settings.json is in sync"
    )
    args = parser.parse_args()

    try:
        if args.validate:
            if validate_sync():
                print(
                    '{"ok": true, "message": "settings.json is in sync with coding-standards.yaml"}'
                )
                sys.exit(0)
            else:
                print(
                    '{"ok": false, "message": "settings.json out of sync. Run: python .claude/hooks/generate_settings.py"}'
                )
                sys.exit(1)

        result = generate_settings(dry_run=args.dry_run, output=args.output)

        if args.dry_run:
            print(json.dumps(result, indent=2))
        else:
            print('{"ok": true, "message": "Settings generated successfully"}')

    except Exception as e:
        logger.exception("Failed to generate settings")
        print(json.dumps({"ok": False, "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
