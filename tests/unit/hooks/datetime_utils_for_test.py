"""Test helper module for datetime_utils.

This module imports datetime_utils from hooks/shared for testing purposes.
It's a thin wrapper to avoid import issues in tests.
"""

import sys
from pathlib import Path

# Add hooks/shared to path
HOOKS_SHARED_PATH = Path(__file__).parent.parent.parent.parent / ".claude" / "hooks"
if str(HOOKS_SHARED_PATH) not in sys.path:
    sys.path.insert(0, str(HOOKS_SHARED_PATH))

# Import all datetime utilities
from shared.datetime_utils import (
    age_in_days,
    age_in_hours,
    format_age,
    get_file_age_days,
    get_file_age_hours,
    is_file_stale,
    is_stale,
    parse_timestamp,
    timestamp_ago,
    timestamp_now,
)

__all__ = [
    "age_in_days",
    "age_in_hours",
    "format_age",
    "get_file_age_days",
    "get_file_age_hours",
    "is_file_stale",
    "is_stale",
    "parse_timestamp",
    "timestamp_ago",
    "timestamp_now",
]
