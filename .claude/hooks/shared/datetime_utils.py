"""Datetime utilities for Claude Code hooks.

Provides standardized datetime operations used across multiple hooks:
- Staleness checks
- Age calculations
- Timestamp parsing

Version: 1.0.0
Created: 2026-01-19
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path


def parse_timestamp(timestamp: str | None) -> datetime | None:
    """Parse ISO timestamp string to datetime object.

    Args:
        timestamp: ISO 8601 timestamp string (may have 'Z' suffix)

    Returns:
        datetime object or None if parsing fails
    """
    if not timestamp:
        return None

    try:
        # Handle 'Z' suffix (common in ISO timestamps)
        cleaned = timestamp.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except (ValueError, TypeError, AttributeError):
        return None


def age_in_hours(timestamp: str | datetime | None) -> float | None:
    """Calculate age of timestamp in hours.

    Args:
        timestamp: ISO timestamp string or datetime object

    Returns:
        Age in hours or None if invalid timestamp
    """
    if timestamp is None:
        return None

    if isinstance(timestamp, str):
        dt = parse_timestamp(timestamp)
        if dt is None:
            return None
    else:
        dt = timestamp

    # Use aware datetime for comparison
    now = datetime.now(UTC) if dt.tzinfo else datetime.now()
    age = now - dt
    return age.total_seconds() / 3600


def age_in_days(timestamp: str | datetime | None) -> int | None:
    """Calculate age of timestamp in days.

    Args:
        timestamp: ISO timestamp string or datetime object

    Returns:
        Age in days or None if invalid timestamp
    """
    hours = age_in_hours(timestamp)
    if hours is None:
        return None
    return int(hours / 24)


def is_stale(
    timestamp: str | datetime | None,
    threshold_hours: int | float,
    fallback_path: Path | None = None,
) -> bool:
    """Check if timestamp is stale (older than threshold).

    Args:
        timestamp: ISO timestamp string or datetime object
        threshold_hours: Threshold in hours for staleness
        fallback_path: Optional file path to check mtime if timestamp is None

    Returns:
        True if stale (older than threshold), False otherwise
    """
    # Try to get age from timestamp
    hours = age_in_hours(timestamp)

    if hours is None and fallback_path:
        # Fallback to file modification time
        try:
            mtime = datetime.fromtimestamp(fallback_path.stat().st_mtime)
            hours = age_in_hours(mtime)
        except (OSError, ValueError):
            # Can't determine age - assume not stale (fail-safe)
            return False

    if hours is None:
        # No timestamp and no fallback - assume not stale (fail-safe)
        return False

    return hours > threshold_hours


def get_file_age_hours(file_path: Path) -> float | None:
    """Get age of file in hours from modification time.

    Args:
        file_path: Path to file

    Returns:
        Age in hours or None if file doesn't exist or error
    """
    try:
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        return age_in_hours(mtime)
    except (OSError, ValueError):
        return None


def get_file_age_days(file_path: Path) -> int | None:
    """Get age of file in days from modification time.

    Args:
        file_path: Path to file

    Returns:
        Age in days or None if file doesn't exist or error
    """
    hours = get_file_age_hours(file_path)
    if hours is None:
        return None
    return int(hours / 24)


def is_file_stale(file_path: Path, threshold_hours: int | float) -> bool:
    """Check if file is stale based on modification time.

    Args:
        file_path: Path to file
        threshold_hours: Threshold in hours for staleness

    Returns:
        True if stale, False if not stale or file doesn't exist
    """
    hours = get_file_age_hours(file_path)
    if hours is None:
        return False
    return hours > threshold_hours


def timestamp_now() -> str:
    """Get current timestamp in ISO 8601 format with UTC timezone.

    Returns:
        ISO 8601 timestamp string
    """
    return datetime.now(UTC).isoformat()


def timestamp_ago(hours: int | float = 0, days: int = 0) -> str:
    """Get timestamp N hours/days ago in ISO 8601 format.

    Args:
        hours: Hours ago
        days: Days ago

    Returns:
        ISO 8601 timestamp string
    """
    delta = timedelta(hours=hours, days=days)
    return (datetime.now(UTC) - delta).isoformat()


def format_age(timestamp: str | datetime | None) -> str:
    """Format age of timestamp as human-readable string.

    Args:
        timestamp: ISO timestamp string or datetime object

    Returns:
        Human-readable age string like "2h ago", "3d ago", "just now"
    """
    hours = age_in_hours(timestamp)
    if hours is None:
        return "unknown age"

    if hours < 1:
        minutes = int(hours * 60)
        if minutes < 1:
            return "just now"
        return f"{minutes}m ago"
    elif hours < 24:
        return f"{int(hours)}h ago"
    else:
        days = int(hours / 24)
        return f"{days}d ago"
