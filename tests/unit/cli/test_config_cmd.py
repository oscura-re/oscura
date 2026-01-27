"""Comprehensive unit tests for config_cmd.py CLI module.

This module provides extensive testing for the Oscura config command, including:
- Configuration file management
- Viewing configuration
- Setting configuration values
- Editing configuration files
- Initializing default configuration
- Path resolution and creation

Test Coverage:
- config() CLI command with all options
- _get_config_path() path resolution
- _initialize_config() default config creation
- _show_config() configuration display
- _set_config_value() value setting with type parsing
- _edit_config() editor integration
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from oscura.cli.config_cmd import (
    _edit_config,
    _get_config_path,
    _initialize_config,
    _set_config_value,
    _show_config,
    config,
)

pytestmark = [pytest.mark.unit, pytest.mark.cli]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cli_runner():
    """Create Click CLI test runner."""
    return CliRunner()


# =============================================================================
# Test _get_config_path()
# =============================================================================


@pytest.mark.unit
def test_get_config_path_local_exists(tmp_path, monkeypatch):
    """Test config path resolution when local config exists."""
    monkeypatch.chdir(tmp_path)
    local_config = tmp_path / ".oscura.yaml"
    local_config.touch()

    result = _get_config_path()

    assert result == local_config


@pytest.mark.unit
def test_get_config_path_user_config(tmp_path, monkeypatch):
    """Test config path resolution falls back to user config."""
    monkeypatch.chdir(tmp_path)

    with patch("pathlib.Path.home") as mock_home:
        mock_home.return_value = tmp_path

        result = _get_config_path()

        expected = tmp_path / ".config" / "oscura" / "config.yaml"
        assert result == expected


@pytest.mark.unit
def test_get_config_path_prefers_local(tmp_path, monkeypatch):
    """Test that local config is preferred over user config."""
    monkeypatch.chdir(tmp_path)
    local_config = tmp_path / ".oscura.yaml"
    local_config.touch()

    with patch("pathlib.Path.home") as mock_home:
        mock_home.return_value = tmp_path
        user_config = tmp_path / ".config" / "oscura" / "config.yaml"
        user_config.parent.mkdir(parents=True, exist_ok=True)
        user_config.touch()

        result = _get_config_path()

        # Should prefer local
        assert result == local_config


# =============================================================================
# Test _initialize_config()
# =============================================================================


@pytest.mark.unit
def test_initialize_config_creates_file(tmp_path):
    """Test that initialize creates config file."""
    config_path = tmp_path / "config.yaml"

    _initialize_config(config_path)

    assert config_path.exists()


@pytest.mark.unit
def test_initialize_config_creates_directories(tmp_path):
    """Test that initialize creates parent directories."""
    config_path = tmp_path / "nested" / "dir" / "config.yaml"

    _initialize_config(config_path)

    assert config_path.parent.exists()
    assert config_path.exists()


@pytest.mark.unit
def test_initialize_config_content(tmp_path):
    """Test that initialized config has expected content."""
    config_path = tmp_path / "config.yaml"

    _initialize_config(config_path)

    content = config_path.read_text()
    assert "analysis:" in content
    assert "export:" in content
    assert "visualization:" in content
    assert "cli:" in content
    assert "logging:" in content


@pytest.mark.unit
def test_initialize_config_default_values(tmp_path):
    """Test that initialized config has sensible defaults."""
    config_path = tmp_path / "config.yaml"

    _initialize_config(config_path)

    content = config_path.read_text()
    assert "default_protocol: auto" in content
    assert "default_format: json" in content
    assert "color_output: true" in content


# =============================================================================
# Test _show_config()
# =============================================================================


@pytest.mark.unit
def test_show_config_displays_content(tmp_path):
    """Test showing existing configuration."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("analysis:\n  default_protocol: uart\n")

    with patch("click.echo") as mock_echo:
        _show_config(config_path)

        # Should display config with path and content
        calls = [str(call) for call in mock_echo.call_args_list]
        output = "".join(calls)
        assert str(config_path) in output
        assert "analysis:" in output
        assert "default_protocol" in output


@pytest.mark.unit
def test_show_config_nonexistent_file(tmp_path):
    """Test showing config when file doesn't exist."""
    config_path = tmp_path / "nonexistent.yaml"

    with patch("click.echo") as mock_echo:
        _show_config(config_path)

        calls = [str(call[0][0]) for call in mock_echo.call_args_list if call[0]]
        output = " ".join(calls)
        assert "No configuration file found" in output


@pytest.mark.unit
def test_show_config_uses_yaml_dump(tmp_path):
    """Test that show_config uses yaml.dump for formatting."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test: value\n")

    with patch("yaml.safe_load") as mock_load:
        with patch("yaml.dump") as mock_dump:
            with patch("click.echo"):
                mock_load.return_value = {"test": "value"}
                mock_dump.return_value = "test: value\n"

                _show_config(config_path)

                mock_dump.assert_called_once()


# =============================================================================
# Test _set_config_value()
# =============================================================================


@pytest.mark.unit
def test_set_config_value_basic(tmp_path):
    """Test setting a simple configuration value."""
    config_path = tmp_path / "config.yaml"

    _set_config_value(config_path, "analysis.default_protocol=uart")

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config["analysis"]["default_protocol"] == "uart"


@pytest.mark.unit
def test_set_config_value_creates_nested_keys(tmp_path):
    """Test that set_config creates nested dictionary structure."""
    config_path = tmp_path / "config.yaml"

    _set_config_value(config_path, "a.b.c.d=value")

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config["a"]["b"]["c"]["d"] == "value"


@pytest.mark.unit
def test_set_config_value_integer(tmp_path):
    """Test setting integer value."""
    config_path = tmp_path / "config.yaml"

    _set_config_value(config_path, "analysis.max_packets=1000")

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config["analysis"]["max_packets"] == 1000
    assert isinstance(config["analysis"]["max_packets"], int)


@pytest.mark.unit
def test_set_config_value_float(tmp_path):
    """Test setting float value."""
    config_path = tmp_path / "config.yaml"

    _set_config_value(config_path, "analysis.threshold=0.75")

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config["analysis"]["threshold"] == 0.75
    assert isinstance(config["analysis"]["threshold"], float)


@pytest.mark.unit
def test_set_config_value_boolean_true(tmp_path):
    """Test setting boolean true value."""
    config_path = tmp_path / "config.yaml"

    _set_config_value(config_path, "cli.color_output=true")

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config["cli"]["color_output"] is True


@pytest.mark.unit
def test_set_config_value_boolean_false(tmp_path):
    """Test setting boolean false value."""
    config_path = tmp_path / "config.yaml"

    _set_config_value(config_path, "cli.progress_bars=false")

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config["cli"]["progress_bars"] is False


@pytest.mark.unit
def test_set_config_value_string(tmp_path):
    """Test setting string value."""
    config_path = tmp_path / "config.yaml"

    _set_config_value(config_path, "logging.level=DEBUG")

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config["logging"]["level"] == "DEBUG"
    assert isinstance(config["logging"]["level"], str)


@pytest.mark.unit
def test_set_config_value_preserves_existing(tmp_path):
    """Test that setting value preserves other config."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("existing:\n  key: value\n")

    _set_config_value(config_path, "new.key=newvalue")

    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config["existing"]["key"] == "value"
    assert config["new"]["key"] == "newvalue"


@pytest.mark.unit
def test_set_config_value_invalid_format():
    """Test error handling for invalid key=value format."""
    config_path = Path("/tmp/config.yaml")

    with pytest.raises(ValueError, match="Invalid format"):
        _set_config_value(config_path, "no_equals_sign")


@pytest.mark.unit
def test_set_config_value_creates_parent_dirs(tmp_path):
    """Test that set_config creates parent directories."""
    config_path = tmp_path / "nested" / "config.yaml"

    _set_config_value(config_path, "key=value")

    assert config_path.parent.exists()
    assert config_path.exists()


# =============================================================================
# Test _edit_config()
# =============================================================================


@pytest.mark.unit
def test_edit_config_uses_editor(tmp_path):
    """Test that edit_config opens configured editor."""
    config_path = tmp_path / "config.yaml"
    config_path.touch()

    with patch.dict(os.environ, {"EDITOR": "vim"}):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            _edit_config(config_path)

            mock_run.assert_called_once()
            assert "vim" in mock_run.call_args[0][0]
            assert str(config_path) in mock_run.call_args[0][0]


@pytest.mark.unit
def test_edit_config_default_editor(tmp_path):
    """Test that edit_config uses nano as default editor."""
    config_path = tmp_path / "config.yaml"
    config_path.touch()

    with patch.dict(os.environ, {}, clear=True):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            _edit_config(config_path)

            assert "nano" in mock_run.call_args[0][0]


@pytest.mark.unit
def test_edit_config_creates_if_missing(tmp_path):
    """Test that edit_config initializes config if missing."""
    config_path = tmp_path / "config.yaml"

    with patch("subprocess.run") as mock_run:
        with patch("oscura.cli.config_cmd._initialize_config") as mock_init:
            mock_run.return_value = Mock(returncode=0)

            _edit_config(config_path)

            mock_init.assert_called_once_with(config_path)


@pytest.mark.unit
def test_edit_config_editor_failure(tmp_path):
    """Test error handling when editor fails."""
    import subprocess

    config_path = tmp_path / "config.yaml"
    config_path.touch()

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "editor")

        with pytest.raises(RuntimeError, match="Editor failed"):
            _edit_config(config_path)


# =============================================================================
# Test config() CLI command
# =============================================================================


@pytest.mark.unit
def test_config_command_show(cli_runner, tmp_path, monkeypatch):
    """Test config --show command."""
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / ".oscura.yaml"
    config_path.write_text("test: value\n")

    with patch("oscura.cli.config_cmd._show_config") as mock_show:
        result = cli_runner.invoke(config, ["--show"], obj={"verbose": 0})

        assert result.exit_code == 0
        mock_show.assert_called_once()


@pytest.mark.unit
def test_config_command_init(cli_runner, tmp_path, monkeypatch):
    """Test config --init command."""
    monkeypatch.chdir(tmp_path)

    with patch("oscura.cli.config_cmd._initialize_config") as mock_init:
        result = cli_runner.invoke(config, ["--init"], obj={"verbose": 0})

        assert result.exit_code == 0
        mock_init.assert_called_once()
        assert "Initialized configuration" in result.output


@pytest.mark.unit
def test_config_command_set(cli_runner, tmp_path, monkeypatch):
    """Test config --set command."""
    monkeypatch.chdir(tmp_path)

    with patch("oscura.cli.config_cmd._set_config_value") as mock_set:
        result = cli_runner.invoke(config, ["--set", "key=value"], obj={"verbose": 0})

        assert result.exit_code == 0
        mock_set.assert_called_once()
        assert "Updated configuration" in result.output


@pytest.mark.unit
def test_config_command_edit(cli_runner, tmp_path, monkeypatch):
    """Test config --edit command."""
    monkeypatch.chdir(tmp_path)

    with patch("oscura.cli.config_cmd._edit_config") as mock_edit:
        result = cli_runner.invoke(config, ["--edit"], obj={"verbose": 0})

        assert result.exit_code == 0
        mock_edit.assert_called_once()


@pytest.mark.unit
def test_config_command_path(cli_runner, tmp_path, monkeypatch):
    """Test config --path command."""
    monkeypatch.chdir(tmp_path)

    result = cli_runner.invoke(config, ["--path"], obj={"verbose": 0})

    assert result.exit_code == 0
    assert "Configuration file:" in result.output


@pytest.mark.unit
def test_config_command_no_options(cli_runner):
    """Test config command with no options shows help."""
    result = cli_runner.invoke(config, [], obj={"verbose": 0})

    assert result.exit_code == 0
    # Should show help text
    assert "Usage:" in result.output or "--help" in result.output


@pytest.mark.unit
def test_config_command_error_handling(cli_runner):
    """Test config command error handling."""
    with patch("oscura.cli.config_cmd._get_config_path") as mock_path:
        mock_path.side_effect = Exception("Failed to get path")

        result = cli_runner.invoke(config, ["--show"], obj={"verbose": 0})

        assert result.exit_code == 1
        assert "Error: Failed to get path" in result.output


@pytest.mark.unit
def test_config_command_error_with_verbose(cli_runner):
    """Test config error handling with verbose mode (should raise)."""
    with patch("oscura.cli.config_cmd._get_config_path") as mock_path:
        mock_path.side_effect = ValueError("Test error")

        result = cli_runner.invoke(config, ["--show"], obj={"verbose": 2})

        assert result.exit_code == 1
        assert isinstance(result.exception, ValueError)


@pytest.mark.unit
def test_config_command_verbose_logging(cli_runner, tmp_path, monkeypatch, caplog):
    """Test config with verbose logging."""
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / ".oscura.yaml"
    config_path.write_text("test: value\n")

    result = cli_runner.invoke(config, ["--show"], obj={"verbose": 1})

    assert result.exit_code == 0


@pytest.mark.unit
def test_config_set_passes_value_correctly(cli_runner, tmp_path, monkeypatch):
    """Test that --set passes value to _set_config_value."""
    monkeypatch.chdir(tmp_path)

    with patch("oscura.cli.config_cmd._set_config_value") as mock_set:
        result = cli_runner.invoke(config, ["--set", "analysis.protocol=uart"], obj={"verbose": 0})

        assert result.exit_code == 0
        call_args = mock_set.call_args
        assert call_args[0][1] == "analysis.protocol=uart"


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
def test_full_config_workflow(cli_runner, tmp_path, monkeypatch):
    """Test complete config workflow: init, set, show."""
    monkeypatch.chdir(tmp_path)

    # Initialize
    result = cli_runner.invoke(config, ["--init"], obj={"verbose": 0})
    assert result.exit_code == 0

    # Set value
    result = cli_runner.invoke(
        config, ["--set", "analysis.default_protocol=uart"], obj={"verbose": 0}
    )
    assert result.exit_code == 0

    # Show config
    result = cli_runner.invoke(config, ["--show"], obj={"verbose": 0})
    assert result.exit_code == 0
    assert "uart" in result.output


@pytest.mark.unit
def test_config_respects_local_vs_user(tmp_path, monkeypatch):
    """Test that config correctly resolves local vs user config."""
    monkeypatch.chdir(tmp_path)

    # No local config
    path1 = _get_config_path()

    # Create local config
    local_config = tmp_path / ".oscura.yaml"
    local_config.touch()
    path2 = _get_config_path()

    # Should prefer local when it exists
    assert path2 == local_config
    assert path1 != path2
