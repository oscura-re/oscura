#!/usr/bin/env python3
"""Tests for enforce_agent_limit.py hook.

Verifies that the agent limit enforcement respects configured limits
correctly and doesn't always default to 2.
"""

import sys
from pathlib import Path

import pytest

# Add hooks directory to path
hooks_dir = Path(__file__).parent.parent.parent / ".claude" / "hooks"
sys.path.insert(0, str(hooks_dir))

from enforce_agent_limit import DEFAULT_MAX_RUNNING, get_max_running_limit

pytestmark = [pytest.mark.unit, pytest.mark.core]


def test_get_max_running_limit_with_max_concurrent_only():
    """With max_concurrent=3, should return 3 (not DEFAULT)."""
    config = {"orchestration": {"agents": {"max_concurrent": 3}}}
    assert get_max_running_limit(config) == 3


def test_get_max_running_limit_with_both_configured():
    """With both max_concurrent=3 and max_batch=5, should return 3 (most restrictive)."""
    config = {"orchestration": {"agents": {"max_concurrent": 3, "max_batch_size": 5}}}
    assert get_max_running_limit(config) == 3


def test_get_max_running_limit_with_batch_more_restrictive():
    """With max_concurrent=5 and max_batch=3, should return 3 (most restrictive)."""
    config = {"orchestration": {"agents": {"max_concurrent": 5, "max_batch_size": 3}}}
    assert get_max_running_limit(config) == 3


def test_get_max_running_limit_with_no_config():
    """With no config, should return DEFAULT_MAX_RUNNING."""
    config = {"orchestration": {"agents": {}}}
    assert get_max_running_limit(config) == DEFAULT_MAX_RUNNING


def test_get_max_running_limit_empty_config():
    """With empty config, should return DEFAULT_MAX_RUNNING."""
    config = {}
    assert get_max_running_limit(config) == DEFAULT_MAX_RUNNING


def test_get_max_running_limit_legacy_swarm_config():
    """Should fallback to legacy swarm config if orchestration not present."""
    config = {"swarm": {"max_parallel_agents": 4}}
    assert get_max_running_limit(config) == 4


def test_get_max_running_limit_legacy_batch_size():
    """Should fallback to legacy swarm batch size if orchestration not present."""
    config = {"swarm": {"max_batch_size": 6}}
    assert get_max_running_limit(config) == 6


def test_bug_scenario_max_concurrent_3():
    """
    REGRESSION TEST: Bug where max_concurrent=3 returned 2.

    Before fix: limits = [2, 3] → min = 2
    After fix: limits = [3] → min = 3
    """
    config = {"orchestration": {"agents": {"max_concurrent": 3, "max_batch_size": 3}}}
    result = get_max_running_limit(config)
    assert result == 3, f"Bug regression: expected 3, got {result}"
