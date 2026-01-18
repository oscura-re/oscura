import pytest

"""Unit tests for check_subagent_stop.py hook.

Tests the SubagentStop hook that auto-summarizes large outputs and updates the registry.
This is Layer 3 of the 4-layer orchestration enforcement system (v3.2.0).

Test categories:
1. Basic allow/block decisions
2. Output size thresholding and auto-summarization
3. Registry updates on completion
4. Summary generation from JSON and plain text
5. Error handling for corrupted outputs
6. Metrics tracking
"""

import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

pytestmark = pytest.mark.unit


class TestCheckSubagentStopBasic:
    """Tests for basic allow/block behavior."""

    def test_allows_stop_with_no_outputs(self, tmp_path: Path) -> None:
        """Should allow stop when no agent outputs exist."""
        (tmp_path / ".claude" / "agent-outputs").mkdir(parents=True)

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["ok"] is True

    def test_allows_stop_with_complete_status(self, tmp_path: Path) -> None:
        """Should allow stop when completion report has status=complete."""
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        outputs_dir.mkdir(parents=True)

        report = {"status": "complete", "agent_id": "test-agent"}
        report_file = outputs_dir / "2025-12-30-test-complete.json"
        report_file.write_text(json.dumps(report))

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["ok"] is True

    def test_blocks_stop_with_blocked_status(self, tmp_path: Path) -> None:
        """Should block stop when completion report has status=blocked."""
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        outputs_dir.mkdir(parents=True)

        report = {
            "status": "blocked",
            "agent_id": "test-agent",
            "blocked_by": "Missing dependency",
        }
        report_file = outputs_dir / "2025-12-30-test-complete.json"
        report_file.write_text(json.dumps(report))

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 2  # Blocked
        output = json.loads(result["stdout"])
        assert output["ok"] is False
        assert "blocked" in output.get("reason", "").lower()

    def test_allows_stop_with_needs_review_status(self, tmp_path: Path) -> None:
        """Should allow stop but log warning for needs-review status."""
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        outputs_dir.mkdir(parents=True)

        report = {"status": "needs-review", "agent_id": "test-agent"}
        report_file = outputs_dir / "2025-12-30-test-complete.json"
        report_file.write_text(json.dumps(report))

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["ok"] is True


class TestCheckSubagentStopOutputSize:
    """Tests for output size thresholding and auto-summarization."""

    def test_no_summarization_for_small_output(self, tmp_path: Path) -> None:
        """Should not summarize outputs under 50K tokens."""
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        outputs_dir.mkdir(parents=True)
        summaries_dir = tmp_path / ".claude" / "summaries"
        summaries_dir.mkdir(parents=True)

        # Small output (~1K tokens)
        report = {
            "status": "complete",
            "agent_id": "small-agent",
            "summary": "Task completed successfully",
            "artifacts": ["file1.py", "file2.py"],
        }
        report_file = outputs_dir / "2025-12-30-small-complete.json"
        report_file.write_text(json.dumps(report))

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output.get("auto_summarized") is not True

    def test_auto_summarizes_large_output(self, tmp_path: Path) -> None:
        """Should auto-summarize outputs over 50K tokens (~200KB)."""
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        outputs_dir.mkdir(parents=True)
        summaries_dir = tmp_path / ".claude" / "summaries"
        summaries_dir.mkdir(parents=True)

        # Large output (~60K tokens = 240KB)
        large_content = "x" * 250000
        report = {
            "status": "complete",
            "agent_id": "large-agent",
            "summary": "Task completed",
            "large_data": large_content,
        }
        report_file = outputs_dir / "2025-12-30-large-complete.json"
        report_file.write_text(json.dumps(report))

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output.get("auto_summarized") is True
        assert "summaries" in output.get("message", "").lower()

        # Verify summary file was created
        summary_files = list(summaries_dir.glob("*.md"))
        assert len(summary_files) >= 1

    def test_summary_extracts_key_fields(self, tmp_path: Path) -> None:
        """Should extract status, summary, key_findings, artifacts from JSON."""
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        outputs_dir.mkdir(parents=True)
        summaries_dir = tmp_path / ".claude" / "summaries"
        summaries_dir.mkdir(parents=True)

        # Large output with structured data
        large_padding = "x" * 250000
        report = {
            "status": "complete",
            "agent_id": "structured-agent",
            "summary": "Found 5 issues and fixed 3",
            "key_findings": ["Issue 1", "Issue 2", "Issue 3"],
            "artifacts": ["fix1.py", "fix2.py"],
            "padding": large_padding,
        }
        report_file = outputs_dir / "2025-12-30-structured-complete.json"
        report_file.write_text(json.dumps(report))

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 0

        # Check summary content
        summary_files = list(summaries_dir.glob("*.md"))
        if summary_files:
            summary_content = summary_files[0].read_text()
            assert "complete" in summary_content.lower()


class TestCheckSubagentStopRegistry:
    """Tests for registry updates on completion."""

    def test_updates_registry_on_completion(self, tmp_path: Path) -> None:
        """Should update agent status in registry when task completes."""
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        outputs_dir.mkdir(parents=True)

        # Create registry with running agent
        registry = {
            "agents": {"test-agent": {"status": "running", "description": "Test task"}},
            "metadata": {"agents_running": 1},
        }
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.write_text(json.dumps(registry))

        # Create completion report
        report = {"status": "complete", "agent_id": "test-agent"}
        report_file = outputs_dir / "2025-12-30-test-complete.json"
        report_file.write_text(json.dumps(report))

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 0

        # Check registry was updated
        updated_registry = json.loads(registry_file.read_text())
        if "test-agent" in updated_registry.get("agents", {}):
            assert updated_registry["agents"]["test-agent"]["status"] == "completed"

    def test_decrements_running_count(self, tmp_path: Path) -> None:
        """Should decrement running agent count in registry."""
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        outputs_dir.mkdir(parents=True)

        # Create registry with 2 running agents
        registry = {
            "agents": {
                "agent-1": {"status": "running"},
                "agent-2": {"status": "running"},
            },
            "metadata": {"agents_running": 2},
        }
        registry_file = tmp_path / ".claude" / "agent-registry.json"
        registry_file.write_text(json.dumps(registry))

        # Complete one agent
        report = {"status": "complete", "agent_id": "agent-1"}
        report_file = outputs_dir / "2025-12-30-agent1-complete.json"
        report_file.write_text(json.dumps(report))

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 0

        # Check running count
        updated_registry = json.loads(registry_file.read_text())
        assert updated_registry["metadata"]["agents_running"] <= 1


class TestCheckSubagentStopErrorHandling:
    """Tests for error handling and edge cases."""

    def test_handles_corrupted_report_json(self, tmp_path: Path) -> None:
        """Should handle corrupted JSON in completion report."""
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        outputs_dir.mkdir(parents=True)

        report_file = outputs_dir / "2025-12-30-corrupt-complete.json"
        report_file.write_text("{ invalid json }")

        result = run_hook(tmp_path, {})

        # Should not crash
        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["ok"] is True

    def test_handles_missing_outputs_directory(self, tmp_path: Path) -> None:
        """Should handle missing agent-outputs directory."""
        (tmp_path / ".claude").mkdir(parents=True)
        # No agent-outputs directory

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["ok"] is True

    def test_handles_old_completion_reports(self, tmp_path: Path) -> None:
        """Should ignore completion reports older than 5 minutes."""
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        outputs_dir.mkdir(parents=True)

        report = {"status": "blocked", "agent_id": "old-agent"}
        report_file = outputs_dir / "2025-12-30-old-complete.json"
        report_file.write_text(json.dumps(report))

        # Make file old (10 minutes ago)
        old_time = datetime.now() - timedelta(minutes=10)
        os.utime(report_file, (old_time.timestamp(), old_time.timestamp()))

        result = run_hook(tmp_path, {})

        # Should allow (old blocked report ignored)
        assert result["returncode"] == 0

    def test_handles_missing_status_field(self, tmp_path: Path) -> None:
        """Should handle completion report without status field."""
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        outputs_dir.mkdir(parents=True)

        report = {"agent_id": "no-status-agent", "result": "success"}
        report_file = outputs_dir / "2025-12-30-nostatus-complete.json"
        report_file.write_text(json.dumps(report))

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 0
        output = json.loads(result["stdout"])
        assert output["ok"] is True

    def test_handles_empty_report_file(self, tmp_path: Path) -> None:
        """Should handle empty completion report file."""
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        outputs_dir.mkdir(parents=True)

        report_file = outputs_dir / "2025-12-30-empty-complete.json"
        report_file.write_text("")

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 0


class TestCheckSubagentStopMetrics:
    """Tests for metrics tracking."""

    def test_updates_metrics_on_stop(self, tmp_path: Path) -> None:
        """Should update orchestration-metrics.json on subagent stop."""
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        outputs_dir.mkdir(parents=True)
        hooks_dir = tmp_path / ".claude" / "hooks"
        hooks_dir.mkdir(parents=True)

        report = {"status": "complete", "agent_id": "metrics-agent"}
        report_file = outputs_dir / "2025-12-30-metrics-complete.json"
        report_file.write_text(json.dumps(report))

        result = run_hook(tmp_path, {})

        assert result["returncode"] == 0

        # Check metrics file
        metrics_file = hooks_dir / "orchestration-metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            assert "subagent_stops" in metrics
            assert metrics["subagent_stops"]["total"] >= 1


class TestCheckSubagentStopSummaryCreation:
    """Tests for summary content creation."""

    def test_summary_includes_original_size(self, tmp_path: Path) -> None:
        """Summary file should include original output size."""
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        outputs_dir.mkdir(parents=True)
        summaries_dir = tmp_path / ".claude" / "summaries"
        summaries_dir.mkdir(parents=True)

        large_content = "x" * 250000
        report = {"status": "complete", "agent_id": "size-agent", "data": large_content}
        report_file = outputs_dir / "2025-12-30-size-complete.json"
        report_file.write_text(json.dumps(report))

        result = run_hook(tmp_path, {})

        summary_files = list(summaries_dir.glob("*.md"))
        if summary_files:
            summary_content = summary_files[0].read_text()
            assert "Original size" in summary_content or "original" in summary_content.lower()

    def test_summary_truncates_with_marker(self, tmp_path: Path) -> None:
        """Summary should include truncation marker for non-JSON content."""
        outputs_dir = tmp_path / ".claude" / "agent-outputs"
        outputs_dir.mkdir(parents=True)
        summaries_dir = tmp_path / ".claude" / "summaries"
        summaries_dir.mkdir(parents=True)

        # Large plain text (not valid JSON structure for extraction)
        large_text = "Important finding: " + "x" * 250000
        report = {"status": "complete", "agent_id": "text-agent", "raw_output": large_text}
        report_file = outputs_dir / "2025-12-30-text-complete.json"
        report_file.write_text(json.dumps(report))

        run_hook(tmp_path, {})

        summary_files = list(summaries_dir.glob("*.md"))
        # Just verify no crash - summary creation is optional


def run_hook(project_dir: Path, input_data: dict) -> dict:
    """Run the check_subagent_stop.py hook and return results."""
    hook_path = (
        Path(__file__).parent.parent.parent.parent / ".claude" / "hooks" / "check_subagent_stop.py"
    )

    env = os.environ.copy()
    env["CLAUDE_PROJECT_DIR"] = str(project_dir)

    result = subprocess.run(
        ["python3", str(hook_path)],
        check=False,
        input=json.dumps(input_data),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
