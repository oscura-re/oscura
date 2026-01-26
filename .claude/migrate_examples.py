#!/usr/bin/env python3
"""Migrate examples from demos/ to demonstrations/.

Implements the consolidation plan to create a single optimal examples directory.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
PLAN_FILE = ROOT / ".claude" / "optimal_structure.json"


def load_plan() -> dict:
    """Load implementation plan."""
    with open(PLAN_FILE) as f:
        return json.load(f)


def update_imports(content: str, old_path: str, new_path: str) -> str:
    """Update import paths in migrated file.

    Args:
        content: File content
        old_path: Original path (e.g., demos/04_serial_protocols/i2s_demo.py)
        new_path: New path (e.g., demonstrations/03_protocol_decoding/i2s.py)

    Returns:
        Updated content
    """
    # Update common imports
    content = content.replace("from demos.common", "from demonstrations.common")
    content = content.replace("import demos.common", "import demonstrations.common")

    # Update data_generation imports
    content = content.replace("from demos.data_generation", "from demonstrations.common")

    # Update demo_data paths to data paths
    content = re.sub(r'demo_data["\']', 'data"', content)

    # Add SKIP_VALIDATION marker if file had it
    if "# SKIP_VALIDATION" in content and not content.startswith('#!/usr/bin/env python3\n"""'):
        # File already has marker, ensure it's in the right place (top of docstring)
        pass

    return content


def migrate_file(source: Path, target: Path, dry_run: bool = False) -> bool:
    """Migrate a single file.

    Args:
        source: Source file path
        target: Target file path
        dry_run: If True, don't actually copy

    Returns:
        True if successful
    """
    try:
        if not source.exists():
            print(f"  ✗ Source not found: {source}")
            return False

        # Read and update content
        content = source.read_text()
        updated_content = update_imports(content, str(source), str(target))

        if dry_run:
            print(f"  ✓ Would migrate: {source.name}")
            return True

        # Create target directory if needed
        target.parent.mkdir(parents=True, exist_ok=True)

        # Write updated content
        target.write_text(updated_content)
        print(f"  ✓ Migrated: {source.name} → {target.name}")
        return True

    except Exception as e:
        print(f"  ✗ Error migrating {source.name}: {e}")
        return False


def main(dry_run: bool = False) -> None:
    """Execute migration.

    Args:
        dry_run: If True, don't actually migrate files
    """
    plan = load_plan()

    print("=" * 80)
    print("EXAMPLE MIGRATION")
    print("=" * 80)
    print()

    if dry_run:
        print("DRY RUN MODE - No files will be modified")
        print()

    # Phase 2: Migrate from demos/
    migrations = plan["implementation_plan"]["phase_2_migrate_from_demos"]

    print(f"Migrating {len(migrations)} files from demos/ to demonstrations/")
    print()

    successful = 0
    failed = 0

    for migration in migrations:
        source_path = ROOT / migration["source"]
        target_path = ROOT / migration["target"]

        print(f"Migrating: {migration['source']}")
        print(f"      to: {migration['target']}")

        if migrate_file(source_path, target_path, dry_run=dry_run):
            successful += 1
        else:
            failed += 1

        print()

    print("=" * 80)
    print("MIGRATION SUMMARY")
    print("=" * 80)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()

    if not dry_run and failed == 0:
        print("✓ All files migrated successfully")
        print()
        print("Next steps:")
        print("1. Run: python demonstrations/validate_all.py")
        print("2. Review any failures")
        print("3. Fix import paths if needed")
        print("4. Run cleanup phase")
    elif dry_run:
        print("Rerun without --dry-run to perform migration")

    print()


if __name__ == "__main__":
    import sys

    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    main(dry_run=dry_run)
