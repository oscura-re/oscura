#!/usr/bin/env python3
"""
Performance Benchmark Tests

Tests hook execution time limits, cleanup performance,
validation speed, and scalability.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest

# Mark all tests in this module as stress tests
pytestmark = pytest.mark.stress

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HOOKS_DIR = PROJECT_ROOT / ".claude" / "hooks"

# Performance thresholds (milliseconds)
MAX_HOOK_EXECUTION_MS = 10000  # 10 seconds
MAX_VALIDATION_MS = 5000  # 5 seconds
MAX_CLEANUP_MS = 30000  # 30 seconds


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_perf_dir() -> Generator[Path, None, None]:
    """Create temporary directory for performance tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Hook Execution Time Tests
# =============================================================================


class TestHookExecutionTime:
    """Test hook execution time limits."""

    @pytest.mark.parametrize(
        "hook_name,max_ms",
        [
            ("check_dependencies.sh", 5000),
            ("check_context_usage.sh", 5000),
            ("post_compact_recovery.sh", 5000),
        ],
    )
    def test_hook_execution_time(self, hook_name: str, max_ms: int) -> None:
        """Test individual hook execution time."""
        hook_path = HOOKS_DIR / hook_name
        if not hook_path.exists():
            pytest.skip(f"Hook not found: {hook_name}")

        start = time.time()
        result = subprocess.run(
            [str(hook_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=max_ms / 1000 + 5,  # Add buffer for timeout
            env={**os.environ, "CLAUDE_PROJECT_DIR": str(PROJECT_ROOT)},
        )
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < max_ms, f"{hook_name} took {elapsed_ms:.0f}ms (max: {max_ms}ms)"

    def test_all_hooks_under_threshold(self) -> None:
        """Test all hooks complete under threshold."""
        hooks = list(HOOKS_DIR.glob("*.sh")) + list(HOOKS_DIR.glob("*.py"))
        slow_hooks = []

        for hook in hooks:
            if hook.name.startswith("_") or hook.name.startswith("."):
                continue
            # Skip test/stress test scripts
            if "test" in hook.name.lower() or "stress" in hook.name.lower():
                continue

            if hook.suffix == ".py":
                cmd = ["uv", "run", "python", str(hook), "--help"]
            else:
                cmd = [str(hook)]

            try:
                start = time.time()
                subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    timeout=MAX_HOOK_EXECUTION_MS / 1000,
                    env={**os.environ, "CLAUDE_PROJECT_DIR": str(PROJECT_ROOT)},
                )
                elapsed_ms = (time.time() - start) * 1000

                if elapsed_ms > MAX_HOOK_EXECUTION_MS:
                    slow_hooks.append((hook.name, elapsed_ms))
            except subprocess.TimeoutExpired:
                slow_hooks.append((hook.name, MAX_HOOK_EXECUTION_MS))
            except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as e:
                # Skip hooks that can't run (missing dependencies, etc.)
                pass

        assert len(slow_hooks) == 0, f"Slow hooks: {slow_hooks}"


# =============================================================================
# Cleanup Performance Tests
# =============================================================================


class TestCleanupPerformance:
    """Test cleanup script performance."""

    def test_pre_compact_cleanup_speed(self) -> None:
        """Test pre_compact_cleanup.sh execution speed."""
        hook_path = HOOKS_DIR / "pre_compact_cleanup.sh"
        if not hook_path.exists():
            pytest.skip("pre_compact_cleanup.sh not found")

        start = time.time()
        result = subprocess.run(
            [str(hook_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=MAX_CLEANUP_MS / 1000,
            env={**os.environ, "CLAUDE_PROJECT_DIR": str(PROJECT_ROOT)},
        )
        elapsed_ms = (time.time() - start) * 1000

        assert result.returncode == 0, f"Cleanup failed: {result.stderr}"
        assert elapsed_ms < MAX_CLEANUP_MS, f"Cleanup took {elapsed_ms:.0f}ms"

    def test_session_cleanup_speed(self) -> None:
        """Test session_cleanup.sh execution speed."""
        hook_path = HOOKS_DIR / "session_cleanup.sh"
        if not hook_path.exists():
            pytest.skip("session_cleanup.sh not found")

        start = time.time()
        result = subprocess.run(
            [str(hook_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=MAX_CLEANUP_MS / 1000,
            env={**os.environ, "CLAUDE_PROJECT_DIR": str(PROJECT_ROOT)},
        )
        elapsed_ms = (time.time() - start) * 1000

        assert result.returncode == 0, f"Cleanup failed: {result.stderr}"
        assert elapsed_ms < MAX_CLEANUP_MS, f"Cleanup took {elapsed_ms:.0f}ms"


# =============================================================================
# Validation Speed Tests
# =============================================================================


class TestValidationSpeed:
    """Test validation script performance."""

    def test_incomplete_features_validation_speed(self) -> None:
        """Test validate_incomplete_features.py speed."""
        hook_path = HOOKS_DIR / "validate_incomplete_features.py"
        if not hook_path.exists():
            pytest.skip("validate_incomplete_features.py not found")

        start = time.time()
        result = subprocess.run(
            ["uv", "run", "python", str(hook_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=MAX_VALIDATION_MS / 1000,
            cwd=str(PROJECT_ROOT),
        )
        elapsed_ms = (time.time() - start) * 1000

        # May fail due to missing markers, but should be fast
        assert elapsed_ms < MAX_VALIDATION_MS, f"Validation took {elapsed_ms:.0f}ms"

    def test_settings_generation_speed(self) -> None:
        """Test generate_settings.py speed."""
        hook_path = HOOKS_DIR / "generate_settings.py"
        if not hook_path.exists():
            pytest.skip("generate_settings.py not found")

        start = time.time()
        result = subprocess.run(
            ["uv", "run", "python", str(hook_path), "--dry-run"],
            check=False,
            capture_output=True,
            text=True,
            timeout=MAX_VALIDATION_MS / 1000,
            cwd=str(PROJECT_ROOT),
        )
        elapsed_ms = (time.time() - start) * 1000

        assert result.returncode == 0, f"Generation failed: {result.stderr}"
        assert elapsed_ms < MAX_VALIDATION_MS, f"Generation took {elapsed_ms:.0f}ms"


# =============================================================================
# Scalability Tests
# =============================================================================


class TestScalability:
    """Test scalability with large inputs."""

    def test_registry_with_1000_agents(self, temp_perf_dir: Path) -> None:
        """Test registry operations with 1000 agents."""
        registry_file = temp_perf_dir / "large_registry.json"

        now = datetime.now(UTC)
        registry = {
            "version": "2.0.0",
            "created_at": now.isoformat(),
            "last_updated": now.isoformat(),
            "agents": {},
            "metadata": {"total_agents_launched": 1000},
        }

        # Create 1000 agents
        for i in range(1000):
            registry["agents"][f"agent-{i:04d}"] = {
                "task_description": f"Task {i}",
                "status": "completed" if i % 2 == 0 else "running",
                "launched_at": now.isoformat(),
            }

        # Time write
        start = time.time()
        with open(registry_file, "w") as f:
            json.dump(registry, f)
        write_ms = (time.time() - start) * 1000

        # Time read
        start = time.time()
        with open(registry_file) as f:
            loaded = json.load(f)
        read_ms = (time.time() - start) * 1000

        # Time filter
        start = time.time()
        running = [a for a, d in loaded["agents"].items() if d["status"] == "running"]
        filter_ms = (time.time() - start) * 1000

        assert write_ms < 1000, f"Write took {write_ms:.0f}ms"
        assert read_ms < 500, f"Read took {read_ms:.0f}ms"
        assert filter_ms < 100, f"Filter took {filter_ms:.0f}ms"
        assert len(running) == 500

    def test_large_config_parsing(self, temp_perf_dir: Path) -> None:
        """Test parsing large configuration file."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        config_file = temp_perf_dir / "large_config.yaml"

        # Generate large config
        lines = ["config:"]
        for i in range(1000):
            lines.append(f"  key_{i}: value_{i}")

        config_file.write_text("\n".join(lines))

        # Time parse
        start = time.time()
        with open(config_file) as f:
            data = yaml.safe_load(f)
        parse_ms = (time.time() - start) * 1000

        assert parse_ms < 1000, f"Parse took {parse_ms:.0f}ms"
        assert len(data["config"]) == 1000

    def test_file_scanning_performance(self, temp_perf_dir: Path) -> None:
        """Test file scanning performance."""
        # Create 500 files in nested structure
        for i in range(10):
            subdir = temp_perf_dir / f"dir_{i}"
            subdir.mkdir()
            for j in range(50):
                (subdir / f"file_{j}.py").write_text(f"# File {i}/{j}\n")

        # Time recursive scan
        start = time.time()
        files = list(temp_perf_dir.rglob("*.py"))
        scan_ms = (time.time() - start) * 1000

        assert len(files) == 500
        assert scan_ms < 1000, f"Scan took {scan_ms:.0f}ms"


# =============================================================================
# Memory Tests
# =============================================================================


class TestMemoryUsage:
    """Test memory usage patterns."""

    def test_json_memory_efficient_read(self, temp_perf_dir: Path) -> None:
        """Test memory-efficient JSON reading."""
        # Create 10MB JSON file
        data = {"items": [{"id": i, "data": "x" * 1000} for i in range(10000)]}
        json_file = temp_perf_dir / "large.json"

        with open(json_file, "w") as f:
            json.dump(data, f)

        file_size_mb = json_file.stat().st_size / (1024 * 1024)
        assert file_size_mb >= 9, f"File only {file_size_mb:.1f}MB"

        # Read and verify
        start = time.time()
        with open(json_file) as f:
            loaded = json.load(f)
        read_ms = (time.time() - start) * 1000

        assert len(loaded["items"]) == 10000
        assert read_ms < 5000, f"Read took {read_ms:.0f}ms for {file_size_mb:.1f}MB"


# =============================================================================
# Main
# =============================================================================
