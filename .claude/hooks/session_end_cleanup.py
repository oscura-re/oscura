#!/usr/bin/env python3
"""
Session End Cleanup Hook
Performs comprehensive cleanup tasks when a session ends.
Consolidates functionality from session_cleanup.sh and cleanup_completed_workflows.sh.

Operations:
1. Remove temporary files (*.tmp, *.temp, *.bak, *.backup, *.partial, *.swp, *~)
2. Remove expired lock files (check expires_at field, fallback to mtime)
3. Clean orphaned translation chunks (chunk-* without -translated)
4. Archive workflow-progress.json files

Version: 1.0.0
Created: 2026-01-19
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent))
from shared import get_hook_logger, is_file_stale, load_config, parse_timestamp

# Load configuration
PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
CONFIG = load_config(PROJECT_DIR)

# Get configuration values
RETENTION = CONFIG.get("retention", {})
LOCKS_STALE_MINUTES = RETENTION.get("locks_stale_minutes", 60)

CLEANUP_PATTERNS = CONFIG.get("cleanup", {}).get(
    "temporary_files",
    [
        "*.tmp",
        "*.temp",
        "*.bak",
        "*.backup",
        "*.partial",
        "*.swp",
        "*~",
    ],
)

ORPHAN_CONFIG = CONFIG.get("cleanup", {}).get("orphan_detection", {})
ORPHAN_PATTERN = ORPHAN_CONFIG.get("pattern", "chunk-*.md")
ORPHAN_MISSING = ORPHAN_CONFIG.get("missing_counterpart", "-translated.md")
ORPHAN_AGE_HOURS = ORPHAN_CONFIG.get("age_hours", 24)

# Logger
logger = get_hook_logger(__name__)


def remove_temporary_files(coord_dir: Path) -> int:
    """Remove temporary files matching configured patterns.

    Args:
        coord_dir: Path to .coordination directory

    Returns:
        Number of files removed
    """
    if not coord_dir.exists():
        return 0

    removed_count = 0

    for pattern in CLEANUP_PATTERNS:
        try:
            for temp_file in coord_dir.rglob(pattern):
                if temp_file.is_file():
                    try:
                        temp_file.unlink()
                        removed_count += 1
                    except OSError as e:
                        logger.warning(f"Failed to remove {temp_file}: {e}")
        except Exception as e:
            logger.warning(f"Error processing pattern '{pattern}': {e}")

    return removed_count


def remove_expired_locks(locks_dir: Path) -> int:
    """Remove expired lock files.

    Checks expires_at field in JSON, falls back to modification time.

    Args:
        locks_dir: Path to locks directory

    Returns:
        Number of locks removed
    """
    if not locks_dir.exists():
        return 0

    removed_count = 0
    threshold_hours = LOCKS_STALE_MINUTES / 60

    for lock_file in locks_dir.glob("*.json"):
        if not lock_file.is_file():
            continue

        should_remove = False

        # Try to read expires_at from JSON
        try:
            with lock_file.open() as f:
                lock_data = json.load(f)

            expires_at = lock_data.get("expires_at")
            if expires_at:
                # Check if lock has expired
                expires_dt = parse_timestamp(expires_at)
                if expires_dt and datetime.now() > expires_dt:
                    should_remove = True
                    logger.info(f"Lock expired: {lock_file.name}")
            else:
                # No expires_at field - fallback to mtime
                if is_file_stale(lock_file, threshold_hours):
                    should_remove = True
                    logger.info(f"Lock stale (no expires_at): {lock_file.name}")

        except (OSError, json.JSONDecodeError) as e:
            # Corrupted JSON - use mtime fallback
            logger.warning(f"Failed to read {lock_file.name}: {e}")
            if is_file_stale(lock_file, threshold_hours):
                should_remove = True
                logger.info(f"Lock stale (corrupted): {lock_file.name}")

        if should_remove:
            try:
                lock_file.unlink()
                removed_count += 1
            except OSError as e:
                logger.warning(f"Failed to remove {lock_file}: {e}")

    return removed_count


def clean_orphaned_chunks(translation_dir: Path) -> int:
    """Clean up orphaned translation chunks.

    Removes chunk-*.md files older than configured age that have no
    corresponding -translated.md file.

    Args:
        translation_dir: Path to translation directory

    Returns:
        Number of chunks removed
    """
    if not translation_dir.exists():
        return 0

    removed_count = 0

    # Find all subdirectories in translation/
    for subdir in translation_dir.iterdir():
        if not subdir.is_dir():
            continue

        # Find chunk files in this subdirectory
        for chunk_file in subdir.glob(ORPHAN_PATTERN):
            if not chunk_file.is_file():
                continue

            # Check for corresponding translated file
            translated_file = chunk_file.with_name(
                chunk_file.stem + ORPHAN_MISSING.replace(".md", "") + ".md"
            )

            # If no translated version and chunk is old, remove it
            if not translated_file.exists() and is_file_stale(chunk_file, ORPHAN_AGE_HOURS):
                try:
                    chunk_file.unlink()
                    removed_count += 1
                    logger.info(f"Removed orphaned chunk: {chunk_file.name}")
                except OSError as e:
                    logger.warning(f"Failed to remove {chunk_file}: {e}")

    return removed_count


def archive_workflow_progress(workflow_file: Path) -> int:
    """Archive workflow-progress.json file if it exists.

    Args:
        workflow_file: Path to workflow-progress.json

    Returns:
        Number of files archived (0 or 1)
    """
    if not workflow_file.exists():
        return 0

    try:
        # Generate timestamped archive name
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        archive_name = workflow_file.with_name(f"workflow-progress-archived-{timestamp}.json")

        # Move to archive
        workflow_file.rename(archive_name)
        logger.info(f"Archived workflow progress to {archive_name.name}")
        return 1

    except OSError as e:
        logger.warning(f"Failed to archive workflow progress: {e}")
        return 0


def main() -> None:
    """Main entry point."""
    try:
        coord_dir = PROJECT_DIR / ".coordination"
        locks_dir = coord_dir / "locks"
        translation_dir = coord_dir / "translation"
        workflow_file = PROJECT_DIR / ".claude" / "workflow-progress.json"

        # Perform cleanup operations
        temp_count = remove_temporary_files(coord_dir)
        lock_count = remove_expired_locks(locks_dir)
        chunk_count = clean_orphaned_chunks(translation_dir)
        archived_count = archive_workflow_progress(workflow_file)

        # Log summary
        logger.info(
            f"SessionEnd cleanup complete: "
            f"{temp_count} temp files, "
            f"{lock_count} locks, "
            f"{chunk_count} orphan chunks, "
            f"{archived_count} workflows archived"
        )

        # Return success response
        result = {
            "ok": True,
            "removed_temp_files": temp_count,
            "removed_locks": lock_count,
            "removed_orphan_chunks": chunk_count,
            "archived_workflows": archived_count,
        }
        print(json.dumps(result))
        sys.exit(0)

    except Exception as e:
        logger.error(f"Unexpected error during cleanup: {e}", exc_info=True)
        # Fail gracefully - don't block session end
        print(json.dumps({"ok": True, "error": str(e)}))
        sys.exit(0)


if __name__ == "__main__":
    main()
