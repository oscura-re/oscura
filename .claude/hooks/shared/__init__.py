"""Shared utilities for Claude Code hooks.

This package provides common functionality used across multiple hooks:
- paths.py: Path definitions and access
"""

from .paths import PATHS, get_path, load_paths

__all__ = [
    "PATHS",
    "get_path",
    "load_paths",
]
