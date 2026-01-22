#!/usr/bin/env python3
"""Comprehensive test suite for phantom agent detection and cleanup.

Tests both fix_phantom_agents.py and enhanced cleanup_stale_agents.py.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Add hooks directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules to test
import fix_phantom_agents
from cleanup_stale_agents import (
    get_agent_activity_time,
    is_agent_active,
    is_agent_stale,
)


class TestFixture:
    """Test fixture for phantom agent testing."""

    def __init__(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.project_root = self.test_dir / "project"
        self.project_root.mkdir()

        self.claude_dir = self.project_root / ".claude"
        self.claude_dir.mkdir()

        self.task_dir = Path(tempfile.mkdtemp()) / "tasks"
        self.task_dir.mkdir(parents=True)

        self.registry_path = self.claude_dir / "agent-registry.json"
        self.agent_outputs_dir = self.claude_dir / "agent-outputs"
        self.agent_outputs_dir.mkdir()
        self.summaries_dir = self.claude_dir / "summaries"
        self.summaries_dir.mkdir()

    def cleanup(self):
        """Clean up test directories."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        shutil.rmtree(self.task_dir.parent, ignore_errors=True)

    def create_registry(self, agents: dict) -> None:
        """Create a test registry."""
        registry = {"agents": agents}
        with self.registry_path.open("w") as f:
            json.dump(registry, f, indent=2)

    def create_task_output(self, agent_id: str, content: str = "", age_hours: float = 0) -> Path:
        """Create a task output file."""
        output_file = self.task_dir / f"{agent_id}.output"
        output_file.write_text(content)

        if age_hours > 0:
            # Set modification time to simulate age
            age_time = datetime.now() - timedelta(hours=age_hours)
            import os

            os.utime(output_file, (age_time.timestamp(), age_time.timestamp()))

        return output_file


def test_phantom_detection_missing_file(fixture: TestFixture) -> dict:
    """Test 1: Detect phantom agent with missing output file."""
    print("\n" + "=" * 70)
    print("TEST 1: Phantom Detection - Missing Output File")
    print("=" * 70)

    # Create agent with missing output file
    agents = {
        "phantom1": {
            "type": "test_agent",
            "task": "Test task",
            "status": "running",
            "output_file": str(fixture.task_dir / "phantom1.output"),
            "launched_at": (datetime.now(UTC) - timedelta(hours=25)).isoformat(),
        }
    }
    fixture.create_registry(agents)

    # Run detection (dry run)
    stats = fix_phantom_agents.cleanup_phantom_agents(fixture.registry_path, dry_run=True)

    result = {
        "test": "phantom_detection_missing_file",
        "passed": stats["phantom_agents"] == 1,
        "phantom_count": stats["phantom_agents"],
        "expected": 1,
        "details": stats,
    }

    print(f"✓ Detected {stats['phantom_agents']} phantom agents (expected 1)")
    print(f"  Reason: {stats['fixed'][0]['reason'] if stats['fixed'] else 'N/A'}")

    return result


def test_phantom_detection_old_empty_file(fixture: TestFixture) -> dict:
    """Test 2: Detect phantom agent with old empty output file."""
    print("\n" + "=" * 70)
    print("TEST 2: Phantom Detection - Old Empty File")
    print("=" * 70)

    # Create agent with old empty file
    output_file = fixture.create_task_output("phantom2", content="", age_hours=2)
    agents = {
        "phantom2": {
            "type": "test_agent",
            "task": "Test task",
            "status": "running",
            "output_file": str(output_file),
            "launched_at": (datetime.now(UTC) - timedelta(hours=25)).isoformat(),
        }
    }
    fixture.create_registry(agents)

    # Run detection
    stats = fix_phantom_agents.cleanup_phantom_agents(fixture.registry_path, dry_run=True)

    result = {
        "test": "phantom_detection_old_empty_file",
        "passed": stats["phantom_agents"] == 1,
        "phantom_count": stats["phantom_agents"],
        "expected": 1,
        "details": stats,
    }

    print(f"✓ Detected {stats['phantom_agents']} phantom agents (expected 1)")
    print(f"  File size: 0 bytes, age: 2 hours")

    return result


def test_active_agent_preservation(fixture: TestFixture) -> dict:
    """Test 3: Preserve active agents with recent output."""
    print("\n" + "=" * 70)
    print("TEST 3: Active Agent Preservation")
    print("=" * 70)

    # Create agent with recent output
    output_file = fixture.create_task_output("active1", content="Active output", age_hours=0.5)
    agents = {
        "active1": {
            "type": "test_agent",
            "task": "Test task",
            "status": "running",
            "output_file": str(output_file),
            "launched_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
        }
    }
    fixture.create_registry(agents)

    # Run detection
    stats = fix_phantom_agents.cleanup_phantom_agents(fixture.registry_path, dry_run=True)

    result = {
        "test": "active_agent_preservation",
        "passed": stats["phantom_agents"] == 0,
        "phantom_count": stats["phantom_agents"],
        "expected": 0,
        "details": stats,
    }

    print(f"✓ Preserved active agent (phantom count: {stats['phantom_agents']}, expected 0)")
    print(f"  File has content and is recent (0.5h old)")

    return result


def test_recent_empty_file_preservation(fixture: TestFixture) -> dict:
    """Test 4: Preserve agents with recent empty files (just launched)."""
    print("\n" + "=" * 70)
    print("TEST 4: Recent Empty File Preservation")
    print("=" * 70)

    # Create agent with very recent empty file
    output_file = fixture.create_task_output("recent1", content="", age_hours=0.01)  # ~30 seconds
    agents = {
        "recent1": {
            "type": "test_agent",
            "task": "Test task",
            "status": "running",
            "output_file": str(output_file),
            "launched_at": datetime.now(UTC).isoformat(),
        }
    }
    fixture.create_registry(agents)

    # Run detection
    stats = fix_phantom_agents.cleanup_phantom_agents(fixture.registry_path, dry_run=True)

    result = {
        "test": "recent_empty_file_preservation",
        "passed": stats["phantom_agents"] == 0,
        "phantom_count": stats["phantom_agents"],
        "expected": 0,
        "details": stats,
    }

    print(
        f"✓ Preserved recently launched agent (phantom count: {stats['phantom_agents']}, expected 0)"
    )
    print(f"  Empty file is very recent (~30 seconds old)")

    return result


def test_cleanup_execution(fixture: TestFixture) -> dict:
    """Test 5: Actual cleanup execution (not dry-run)."""
    print("\n" + "=" * 70)
    print("TEST 5: Cleanup Execution")
    print("=" * 70)

    # Create phantom agent
    agents = {
        "phantom3": {
            "type": "test_agent",
            "task": "Test task",
            "status": "running",
            "output_file": str(fixture.task_dir / "phantom3.output"),
            "launched_at": (datetime.now(UTC) - timedelta(hours=25)).isoformat(),
        }
    }
    fixture.create_registry(agents)

    # Run cleanup (not dry-run)
    stats = fix_phantom_agents.cleanup_phantom_agents(fixture.registry_path, dry_run=False)

    # Check registry was updated
    with fixture.registry_path.open() as f:
        updated_registry = json.load(f)

    agent_status = updated_registry["agents"]["phantom3"]["status"]
    has_cleanup_reason = "cleanup_reason" in updated_registry["agents"]["phantom3"]

    result = {
        "test": "cleanup_execution",
        "passed": agent_status == "stale" and has_cleanup_reason,
        "agent_status": agent_status,
        "has_cleanup_metadata": has_cleanup_reason,
        "expected_status": "stale",
        "details": stats,
    }

    print(f"✓ Cleanup executed successfully")
    print(f"  Agent status: {agent_status} (expected: stale)")
    print(f"  Cleanup metadata: {has_cleanup_reason} (expected: True)")
    print(
        f"  Cleanup reason: {updated_registry['agents']['phantom3'].get('cleanup_reason', 'N/A')}"
    )

    return result


def test_registry_validation(fixture: TestFixture) -> dict:
    """Test 6: Registry structure validation."""
    print("\n" + "=" * 70)
    print("TEST 6: Registry Validation")
    print("=" * 70)

    # Create malformed registry
    with fixture.registry_path.open("w") as f:
        json.dump({}, f)  # Missing 'agents' key

    # Run validation
    validation = fix_phantom_agents.validate_registry(fixture.registry_path, dry_run=False)

    # Check it was fixed
    with fixture.registry_path.open() as f:
        fixed_registry = json.load(f)

    result = {
        "test": "registry_validation",
        "passed": "agents" in fixed_registry and len(validation["fixed"]) > 0,
        "has_agents_key": "agents" in fixed_registry,
        "fixes_applied": len(validation["fixed"]),
        "details": validation,
    }

    print(f"✓ Registry validation successful")
    print(f"  Found {len(validation['issues'])} issues")
    print(f"  Applied {len(validation['fixed'])} fixes")
    print(f"  Registry now has 'agents' key: {'agents' in fixed_registry}")

    return result


def test_enhanced_cleanup_hook_phantom_detection(fixture: TestFixture) -> dict:
    """Test 7: Enhanced cleanup hook phantom detection."""
    print("\n" + "=" * 70)
    print("TEST 7: Enhanced Cleanup Hook - Phantom Detection")
    print("=" * 70)

    # Create phantom agent (missing file)
    agents = {
        "hook_phantom1": {
            "type": "test_agent",
            "task": "Test task",
            "status": "running",
            "output_file": str(fixture.task_dir / "hook_phantom1.output"),
            "launched_at": (datetime.now(UTC) - timedelta(hours=25)).isoformat(),
        }
    }

    agent = agents["hook_phantom1"]
    agent_id = "hook_phantom1"

    # Test activity detection (should return None for missing file)
    activity_time = get_agent_activity_time(agent_id, agent)

    # Test stale detection
    is_stale = is_agent_stale(agent, agent_id)

    result = {
        "test": "enhanced_cleanup_hook_phantom_detection",
        "passed": activity_time is None and is_stale,
        "activity_time": activity_time,
        "is_stale": is_stale,
        "expected_activity_time": None,
        "expected_stale": True,
    }

    print(f"✓ Hook phantom detection works")
    print(f"  Activity time: {activity_time} (expected: None)")
    print(f"  Is stale: {is_stale} (expected: True)")

    return result


def test_enhanced_cleanup_hook_active_detection(fixture: TestFixture) -> dict:
    """Test 8: Enhanced cleanup hook active agent detection."""
    print("\n" + "=" * 70)
    print("TEST 8: Enhanced Cleanup Hook - Active Detection")
    print("=" * 70)

    # Create active agent
    output_file = fixture.create_task_output("hook_active1", content="Active", age_hours=0.5)
    agents = {
        "hook_active1": {
            "type": "test_agent",
            "task": "Test task",
            "status": "running",
            "output_file": str(output_file),
            "launched_at": datetime.now(UTC).isoformat(),
        }
    }

    agent = agents["hook_active1"]
    agent_id = "hook_active1"

    # Test activity detection (should return recent time)
    activity_time = get_agent_activity_time(agent_id, agent)

    # Test active detection
    is_active = is_agent_active(agent, agent_id)

    # Test stale detection (should be False)
    is_stale = is_agent_stale(agent, agent_id)

    result = {
        "test": "enhanced_cleanup_hook_active_detection",
        "passed": activity_time is not None and is_active and not is_stale,
        "activity_time": activity_time,
        "is_active": is_active,
        "is_stale": is_stale,
        "expected_active": True,
        "expected_stale": False,
    }

    print(f"✓ Hook active detection works")
    print(f"  Activity time: {activity_time}")
    print(f"  Is active: {is_active} (expected: True)")
    print(f"  Is stale: {is_stale} (expected: False)")

    return result


def test_multiple_agents_mixed_states(fixture: TestFixture) -> dict:
    """Test 9: Multiple agents in different states."""
    print("\n" + "=" * 70)
    print("TEST 9: Multiple Agents - Mixed States")
    print("=" * 70)

    # Create multiple agents in different states
    active_file = fixture.create_task_output("multi_active", content="Active", age_hours=0.5)
    old_empty_file = fixture.create_task_output("multi_phantom", content="", age_hours=2)

    agents = {
        "multi_active": {
            "type": "test_agent",
            "task": "Active task",
            "status": "running",
            "output_file": str(active_file),
            "launched_at": datetime.now(UTC).isoformat(),
        },
        "multi_phantom": {
            "type": "test_agent",
            "task": "Phantom task",
            "status": "running",
            "output_file": str(old_empty_file),
            "launched_at": (datetime.now(UTC) - timedelta(hours=25)).isoformat(),
        },
        "multi_missing": {
            "type": "test_agent",
            "task": "Missing task",
            "status": "running",
            "output_file": str(fixture.task_dir / "multi_missing.output"),
            "launched_at": (datetime.now(UTC) - timedelta(hours=30)).isoformat(),
        },
        "multi_completed": {
            "type": "test_agent",
            "task": "Completed task",
            "status": "completed",
            "output_file": str(fixture.task_dir / "multi_completed.output"),
        },
    }
    fixture.create_registry(agents)

    # Run detection
    stats = fix_phantom_agents.cleanup_phantom_agents(fixture.registry_path, dry_run=True)

    result = {
        "test": "multiple_agents_mixed_states",
        "passed": stats["phantom_agents"] == 2,  # multi_phantom and multi_missing
        "phantom_count": stats["phantom_agents"],
        "expected": 2,
        "total_agents": stats["total_agents"],
        "details": stats,
    }

    print(f"✓ Mixed state detection works")
    print(f"  Total agents: {stats['total_agents']}")
    print(f"  Phantom agents: {stats['phantom_agents']} (expected: 2)")
    print(f"  Phantom IDs: {[a['agent_id'] for a in stats['fixed']]}")

    return result


def test_error_handling_corrupt_registry(fixture: TestFixture) -> dict:
    """Test 10: Error handling with corrupt registry."""
    print("\n" + "=" * 70)
    print("TEST 10: Error Handling - Corrupt Registry")
    print("=" * 70)

    # Create corrupt JSON
    with fixture.registry_path.open("w") as f:
        f.write("{invalid json")

    # Try to load (should handle gracefully)
    try:
        registry = fix_phantom_agents.load_registry(fixture.registry_path)
        handled_gracefully = registry == {"agents": {}}
        error_occurred = False
    except Exception as e:
        handled_gracefully = False
        error_occurred = True
        error_msg = str(e)

    result = {
        "test": "error_handling_corrupt_registry",
        "passed": handled_gracefully and not error_occurred,
        "handled_gracefully": handled_gracefully,
        "error_occurred": error_occurred,
        "expected_behavior": "Return empty registry",
    }

    if handled_gracefully:
        print(f"✓ Corrupt registry handled gracefully")
        print(f"  Returned empty registry as fallback")
    else:
        print(f"✗ Error occurred: {error_msg if error_occurred else 'Unknown'}")

    return result


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("PHANTOM AGENT COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    results = []

    # Test 1: Phantom detection - missing file
    fixture = TestFixture()
    try:
        results.append(test_phantom_detection_missing_file(fixture))
    finally:
        fixture.cleanup()

    # Test 2: Phantom detection - old empty file
    fixture = TestFixture()
    try:
        results.append(test_phantom_detection_old_empty_file(fixture))
    finally:
        fixture.cleanup()

    # Test 3: Active agent preservation
    fixture = TestFixture()
    try:
        results.append(test_active_agent_preservation(fixture))
    finally:
        fixture.cleanup()

    # Test 4: Recent empty file preservation
    fixture = TestFixture()
    try:
        results.append(test_recent_empty_file_preservation(fixture))
    finally:
        fixture.cleanup()

    # Test 5: Cleanup execution
    fixture = TestFixture()
    try:
        results.append(test_cleanup_execution(fixture))
    finally:
        fixture.cleanup()

    # Test 6: Registry validation
    fixture = TestFixture()
    try:
        results.append(test_registry_validation(fixture))
    finally:
        fixture.cleanup()

    # Test 7: Enhanced hook phantom detection
    fixture = TestFixture()
    try:
        results.append(test_enhanced_cleanup_hook_phantom_detection(fixture))
    finally:
        fixture.cleanup()

    # Test 8: Enhanced hook active detection
    fixture = TestFixture()
    try:
        results.append(test_enhanced_cleanup_hook_active_detection(fixture))
    finally:
        fixture.cleanup()

    # Test 9: Multiple agents mixed states
    fixture = TestFixture()
    try:
        results.append(test_multiple_agents_mixed_states(fixture))
    finally:
        fixture.cleanup()

    # Test 10: Error handling
    fixture = TestFixture()
    try:
        results.append(test_error_handling_corrupt_registry(fixture))
    finally:
        fixture.cleanup()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    for i, result in enumerate(results, 1):
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"{i}. {result['test']}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
