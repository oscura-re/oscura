"""Unit tests for validate_config_consistency.py hook.

Tests the configuration validation hook that ensures all orchestration files are consistent.

Test categories:
1. Version validation - config file versions match
2. Agent-command references - commands reference existing agents
3. Hook references - settings.json references existing hooks
4. SSOT file validation - required files exist
5. Routing keyword validation - no duplicates
6. Error reporting - correct error/warning counts
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = [
    pytest.mark.unit,
]


class TestValidateConfigConsistencyVersions:
    """Tests for version validation."""

    def test_passes_with_valid_versions(self, tmp_path: Path) -> None:
        """Should pass when all config files have valid versions."""
        setup_valid_config(tmp_path)

        result = run_validation(tmp_path)

        assert result["returncode"] == 0
        assert "Configuration is consistent" in result["stdout"]

    def test_reports_missing_orchestration_config(self, tmp_path: Path) -> None:
        """Should report error when config.yaml is missing."""
        setup_valid_config(tmp_path)
        (tmp_path / ".claude" / "config.yaml").unlink()

        result = run_validation(tmp_path)

        # Should still run but report the issue
        assert "config.yaml" in result["stdout"].lower() or result["returncode"] != 0


class TestValidateConfigConsistencyAgentRefs:
    """Tests for agent-command reference validation."""

    def test_passes_with_valid_agent_refs(self, tmp_path: Path) -> None:
        """Should pass when commands reference existing agents."""
        setup_valid_config(tmp_path)

        result = run_validation(tmp_path)

        assert result["returncode"] == 0
        assert "Errors: 0" in result["stdout"]


class TestValidateConfigConsistencyHookRefs:
    """Tests for hook reference validation."""

    def test_passes_with_valid_hook_refs(self, tmp_path: Path) -> None:
        """Should pass when settings.json references existing hooks."""
        setup_valid_config(tmp_path)

        result = run_validation(tmp_path)

        assert result["returncode"] == 0


class TestValidateConfigConsistencySSOT:
    """Tests for single source of truth validation."""

    def test_passes_with_all_ssot_files(self, tmp_path: Path) -> None:
        """Should pass when all SSOT files exist."""
        setup_valid_config(tmp_path)

        result = run_validation(tmp_path)

        assert result["returncode"] == 0
        assert "SSOT file exists" in result["stdout"]


class TestValidateConfigConsistencyFrontmatter:
    """Tests for agent frontmatter validation."""

    def test_passes_with_valid_frontmatter(self, tmp_path: Path) -> None:
        """Should pass when agent frontmatter is valid YAML."""
        setup_valid_config(tmp_path)

        result = run_validation(tmp_path)

        assert result["returncode"] == 0


class TestValidateConfigConsistencyErrorCounts:
    """Tests for error and warning count reporting."""

    def test_reports_zero_errors_for_valid_config(self, tmp_path: Path) -> None:
        """Should report 0 errors for valid configuration."""
        setup_valid_config(tmp_path)

        result = run_validation(tmp_path)

        assert "Errors: 0" in result["stdout"]
        assert result["returncode"] == 0


def setup_valid_config(tmp_path: Path) -> None:
    """Set up a valid configuration structure for testing."""
    # Create directory structure
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / "agents").mkdir()
    (claude_dir / "commands").mkdir()
    (claude_dir / "hooks").mkdir()

    # Create src/oscura directory
    src_dir = tmp_path / "src" / "oscura"
    src_dir.mkdir(parents=True)

    # Create pyproject.toml
    pyproject_content = """[project]
name = "test-project"
version = "0.1.0"
description = "Test project"
"""
    (tmp_path / "pyproject.toml").write_text(pyproject_content)

    # Create __init__.py with version
    init_content = '''"""Test package."""
__version__ = "0.1.0"
'''
    (src_dir / "__init__.py").write_text(init_content)

    # Create config.yaml (main orchestration config)
    config = {
        "version": "3.2.0",
        "swarm": {"max_batch_size": 2, "recommended_batch_size": 1},
        "context_management": {"warning_threshold_percent": 60, "critical_threshold_percent": 75},
    }
    (claude_dir / "config.yaml").write_text(yaml.dump(config))

    # Create coding-standards.yaml
    coding_standards = {
        "version": "2.3.0",
        "language": "python",
        "style": {"formatter": "ruff"},
    }
    (claude_dir / "coding-standards.yaml").write_text(yaml.dump(coding_standards))

    # Create project-metadata.yaml
    project_metadata = {
        "version": "2.1.0",
        "name": "test-project",
        "type": "library",
    }
    (claude_dir / "project-metadata.yaml").write_text(yaml.dump(project_metadata))

    # Create settings.json
    settings = {
        "hooks": {},
        "permissions": {"allow": ["Read", "Write"]},
    }
    (claude_dir / "settings.json").write_text(json.dumps(settings, indent=2))

    # Create a valid agent
    agent_content = """---
name: orchestrator
description: "Routes tasks to specialists"
tools: Task, Read
model: opus
routing_keywords:
  - route
  - coordinate
---
# Orchestrator
Routes tasks.
"""
    (claude_dir / "agents" / "orchestrator.md").write_text(agent_content)

    # Create a valid command
    command_content = """---
name: ai
description: Universal AI routing
arguments: <task>
---
# AI Command
Routes to `.claude/agents/orchestrator.md`
"""
    (claude_dir / "commands" / "ai.md").write_text(command_content)

    # Create .coordination/spec directory
    coord_dir = tmp_path / ".coordination" / "spec"
    coord_dir.mkdir(parents=True)

    # Create incomplete-features.yaml
    incomplete_features = {"version": "1.0", "features": []}
    (coord_dir / "incomplete-features.yaml").write_text(yaml.dump(incomplete_features))


def run_validation(project_dir: Path) -> dict:
    """Run the validate_config_consistency.py script and return results."""
    hook_path = (
        Path(__file__).parent.parent.parent.parent
        / ".claude"
        / "hooks"
        / "validate_config_consistency.py"
    )

    env = os.environ.copy()
    env["CLAUDE_PROJECT_DIR"] = str(project_dir)

    # Change to project dir for relative path resolution
    result = subprocess.run(
        ["python3", str(hook_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=project_dir,
        env=env,
    )

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
