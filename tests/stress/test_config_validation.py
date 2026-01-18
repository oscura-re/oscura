#!/usr/bin/env python3
"""
Configuration Validation Stress Tests

Tests edge cases in YAML parsing, invalid configurations,
missing required fields, circular dependencies, and large-scale configs.
"""

import json
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

# Mark all tests in this module as stress tests
pytestmark = pytest.mark.stress

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HOOKS_DIR = PROJECT_ROOT / ".claude" / "hooks"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_config_dir() -> Generator[Path, None, None]:
    """Create temporary directory for config tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_valid_yaml() -> str:
    """Valid YAML configuration."""
    return """
project:
  name: test-project
  version_source: pyproject.toml

hooks:
  pre_commit:
    - name: test-hook
      script: test.sh
      blocking: true
"""


@pytest.fixture
def sample_invalid_yaml() -> str:
    """Invalid YAML (bad indentation)."""
    return """
project:
  name: test-project
   bad_indent: true
  version: 1.0.0
"""


# =============================================================================
# YAML Parsing Edge Cases
# =============================================================================


class TestYAMLParsing:
    """Test YAML parsing edge cases."""

    def test_empty_yaml(self, temp_config_dir: Path) -> None:
        """Test handling of empty YAML file."""
        yaml_file = temp_config_dir / "empty.yaml"
        yaml_file.write_text("")

        # Should not crash
        try:
            import yaml

            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            assert data is None or data == {}
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_yaml_with_only_comments(self, temp_config_dir: Path) -> None:
        """Test YAML with only comments."""
        yaml_file = temp_config_dir / "comments.yaml"
        yaml_file.write_text("# This is a comment\n# Another comment\n")

        try:
            import yaml

            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            assert data is None
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_unicode_in_yaml(self, temp_config_dir: Path) -> None:
        """Test YAML with unicode characters."""
        yaml_file = temp_config_dir / "unicode.yaml"
        yaml_file.write_text(
            """
project:
  name: "测试项目"
  description: "Emoji test 🚀 🎉"
  author: "José García"
"""
        )

        try:
            import yaml

            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            assert data["project"]["name"] == "测试项目"
            assert "🚀" in data["project"]["description"]
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_very_long_lines(self, temp_config_dir: Path) -> None:
        """Test YAML with very long lines."""
        yaml_file = temp_config_dir / "long_lines.yaml"
        long_value = "x" * 10000
        yaml_file.write_text(f'long_key: "{long_value}"\n')

        try:
            import yaml

            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            assert len(data["long_key"]) == 10000
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_deeply_nested_yaml(self, temp_config_dir: Path) -> None:
        """Test deeply nested YAML structure."""
        yaml_file = temp_config_dir / "deep.yaml"

        # Create 50 levels of nesting
        content = "level0:\n"
        for i in range(1, 50):
            content += "  " * i + f"level{i}:\n"
        content += "  " * 50 + "value: deep\n"

        yaml_file.write_text(content)

        try:
            import yaml

            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            # Navigate to deepest level
            current = data
            for i in range(50):
                current = current[f"level{i}"]
            assert current["value"] == "deep"
        except ImportError:
            pytest.skip("PyYAML not installed")


# =============================================================================
# Invalid Configuration Tests
# =============================================================================


class TestInvalidConfigurations:
    """Test handling of invalid configurations."""

    def test_missing_required_fields(self, temp_config_dir: Path) -> None:
        """Test config missing required fields."""
        yaml_file = temp_config_dir / "missing_fields.yaml"
        yaml_file.write_text(
            """
project:
  # Missing 'name' field
  version_source: pyproject.toml
"""
        )

        try:
            import yaml

            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            # Should load but validation should catch missing fields
            assert "name" not in data.get("project", {})
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_invalid_types(self, temp_config_dir: Path) -> None:
        """Test config with wrong types."""
        yaml_file = temp_config_dir / "wrong_types.yaml"
        yaml_file.write_text(
            """
project:
  name: 12345  # Should be string
  hooks:
    pre_commit: "not a list"  # Should be list
"""
        )

        try:
            import yaml

            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            # YAML loads it, validation should catch type errors
            assert isinstance(data["project"]["name"], int)
            assert isinstance(data["project"]["hooks"]["pre_commit"], str)
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_duplicate_keys(self, temp_config_dir: Path) -> None:
        """Test YAML with duplicate keys."""
        yaml_file = temp_config_dir / "duplicate_keys.yaml"
        yaml_file.write_text(
            """
project:
  name: first
  name: second
"""
        )

        try:
            import yaml

            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            # YAML spec: last value wins
            assert data["project"]["name"] == "second"
        except ImportError:
            pytest.skip("PyYAML not installed")


# =============================================================================
# Large Scale Configuration Tests
# =============================================================================


class TestLargeConfigurations:
    """Test handling of large configurations."""

    def test_1000_item_list(self, temp_config_dir: Path) -> None:
        """Test YAML with 1000-item list."""
        yaml_file = temp_config_dir / "large_list.yaml"

        items = [f"  - item_{i}" for i in range(1000)]
        yaml_file.write_text("items:\n" + "\n".join(items))

        try:
            import yaml

            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            assert len(data["items"]) == 1000
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_1000_key_dict(self, temp_config_dir: Path) -> None:
        """Test YAML with 1000 keys."""
        yaml_file = temp_config_dir / "large_dict.yaml"

        lines = ["config:"]
        for i in range(1000):
            lines.append(f"  key_{i}: value_{i}")

        yaml_file.write_text("\n".join(lines))

        try:
            import yaml

            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            assert len(data["config"]) == 1000
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_1mb_yaml_file(self, temp_config_dir: Path) -> None:
        """Test parsing ~1MB YAML file."""
        yaml_file = temp_config_dir / "large_file.yaml"

        # Generate ~1MB of YAML
        lines = ["data:"]
        for i in range(10000):
            lines.append(f"  key_{i}: {'x' * 100}")

        yaml_file.write_text("\n".join(lines))

        file_size = yaml_file.stat().st_size
        assert file_size > 1_000_000, f"File only {file_size} bytes"

        try:
            import yaml

            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            assert len(data["data"]) == 10000
        except ImportError:
            pytest.skip("PyYAML not installed")


# =============================================================================
# Circular Dependency Detection
# =============================================================================


class TestCircularDependencies:
    """Test detection of circular dependencies in configuration."""

    def test_hook_depends_on_self(self, temp_config_dir: Path) -> None:
        """Test hook that depends on itself."""
        config = {
            "hooks": {
                "pre_commit": [
                    {"name": "hook_a", "depends_on": ["hook_a"]},  # Self-reference
                ]
            }
        }

        yaml_file = temp_config_dir / "self_dep.yaml"

        try:
            import yaml

            with open(yaml_file, "w") as f:
                yaml.dump(config, f)

            with open(yaml_file) as f:
                data = yaml.safe_load(f)

            # Check for self-dependency
            hook = data["hooks"]["pre_commit"][0]
            has_self_dep = hook["name"] in hook.get("depends_on", [])
            assert has_self_dep, "Should detect self-dependency"
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_mutual_dependencies(self, temp_config_dir: Path) -> None:
        """Test hooks with mutual dependencies (A->B, B->A)."""
        config = {
            "hooks": {
                "pre_commit": [
                    {"name": "hook_a", "depends_on": ["hook_b"]},
                    {"name": "hook_b", "depends_on": ["hook_a"]},
                ]
            }
        }

        yaml_file = temp_config_dir / "mutual_dep.yaml"

        try:
            import yaml

            with open(yaml_file, "w") as f:
                yaml.dump(config, f)

            with open(yaml_file) as f:
                data = yaml.safe_load(f)

            # Detect circular dependency
            hooks = {h["name"]: h for h in data["hooks"]["pre_commit"]}
            visited: set[str] = set()
            path: set[str] = set()

            def has_cycle(name: str) -> bool:
                if name in path:
                    return True
                if name in visited:
                    return False
                visited.add(name)
                path.add(name)
                for dep in hooks.get(name, {}).get("depends_on", []):
                    if has_cycle(dep):
                        return True
                path.remove(name)
                return False

            has_circular = any(has_cycle(h) for h in hooks)
            assert has_circular, "Should detect circular dependency"
        except ImportError:
            pytest.skip("PyYAML not installed")


# =============================================================================
# JSON Schema Validation
# =============================================================================


class TestJSONSchemaValidation:
    """Test JSON configuration validation."""

    def test_valid_settings_json(self, temp_config_dir: Path) -> None:
        """Test valid settings.json."""
        settings = {
            "model": "sonnet",
            "cleanupPeriodDays": 30,
            "hooks": {"PreCompact": []},
        }

        settings_file = temp_config_dir / "settings.json"
        with open(settings_file, "w") as f:
            json.dump(settings, f)

        with open(settings_file) as f:
            loaded = json.load(f)

        assert loaded["model"] == "sonnet"
        assert loaded["cleanupPeriodDays"] == 30

    def test_corrupted_json(self, temp_config_dir: Path) -> None:
        """Test handling of corrupted JSON."""
        settings_file = temp_config_dir / "settings.json"
        settings_file.write_text('{"model": "sonnet", invalid}')

        with pytest.raises(json.JSONDecodeError), open(settings_file) as f:
            json.load(f)

    def test_empty_json(self, temp_config_dir: Path) -> None:
        """Test handling of empty JSON file."""
        settings_file = temp_config_dir / "settings.json"
        settings_file.write_text("")

        with pytest.raises(json.JSONDecodeError), open(settings_file) as f:
            json.load(f)


# =============================================================================
# Main
# =============================================================================
