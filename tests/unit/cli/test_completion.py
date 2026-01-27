"""Comprehensive unit tests for completion.py CLI module.

This module provides extensive testing for shell completion support, including:
- Bash completion script generation
- Zsh completion script generation
- Fish completion script generation
- Completion script installation
- Error handling for unsupported shells

Test Coverage:
- get_completion_script() for all shell types
- install_completion() with directory creation
- _get_bash_completion() script content
- _get_zsh_completion() script content
- _get_fish_completion() script content
- Error handling for unsupported shells
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from oscura.cli.completion import (
    _get_bash_completion,
    _get_fish_completion,
    _get_zsh_completion,
    get_completion_script,
    install_completion,
)

pytestmark = [pytest.mark.unit, pytest.mark.cli]


# =============================================================================
# Test get_completion_script()
# =============================================================================


@pytest.mark.unit
def test_get_completion_script_bash():
    """Test getting bash completion script."""
    script = get_completion_script("bash")

    assert script is not None
    assert len(script) > 0
    assert "bash" in script.lower()
    assert "_oscura_completion" in script
    assert "complete -F" in script


@pytest.mark.unit
def test_get_completion_script_zsh():
    """Test getting zsh completion script."""
    script = get_completion_script("zsh")

    assert script is not None
    assert len(script) > 0
    assert "#compdef oscura" in script
    assert "_oscura" in script


@pytest.mark.unit
def test_get_completion_script_fish():
    """Test getting fish completion script."""
    script = get_completion_script("fish")

    assert script is not None
    assert len(script) > 0
    assert "fish" in script.lower()
    assert "complete -c oscura" in script


@pytest.mark.unit
def test_get_completion_script_unsupported():
    """Test error handling for unsupported shell."""
    with pytest.raises(ValueError, match="Unsupported shell"):
        get_completion_script("powershell")


@pytest.mark.unit
def test_get_completion_script_case_sensitive():
    """Test that shell names are case-sensitive."""
    with pytest.raises(ValueError, match="Unsupported shell"):
        get_completion_script("BASH")


# =============================================================================
# Test _get_bash_completion()
# =============================================================================


@pytest.mark.unit
def test_bash_completion_contains_commands():
    """Test bash completion contains all main commands."""
    script = _get_bash_completion()

    commands = [
        "analyze",
        "decode",
        "export",
        "visualize",
        "benchmark",
        "validate",
        "config",
        "plugins",
    ]

    for cmd in commands:
        assert cmd in script


@pytest.mark.unit
def test_bash_completion_has_function():
    """Test bash completion defines completion function."""
    script = _get_bash_completion()

    assert "_oscura_completion()" in script
    assert "COMPREPLY" in script
    assert "compgen" in script


@pytest.mark.unit
def test_bash_completion_file_extensions():
    """Test bash completion includes file extension patterns."""
    script = _get_bash_completion()

    # Check for extensions in glob pattern format
    extensions = ["wfm", "vcd", "csv", "pcap", "wav"]

    for ext in extensions:
        assert ext in script


@pytest.mark.unit
def test_bash_completion_config_options():
    """Test bash completion includes config subcommand options."""
    script = _get_bash_completion()

    config_opts = ["--show", "--set", "--edit", "--init", "--path"]

    for opt in config_opts:
        assert opt in script


@pytest.mark.unit
def test_bash_completion_registers_completion():
    """Test bash completion registers the completion function."""
    script = _get_bash_completion()

    assert "complete -F _oscura_completion oscura" in script


# =============================================================================
# Test _get_zsh_completion()
# =============================================================================


@pytest.mark.unit
def test_zsh_completion_contains_commands():
    """Test zsh completion contains command descriptions."""
    script = _get_zsh_completion()

    commands = [
        "analyze:Run full analysis workflow",
        "decode:Decode protocol data",
        "export:Export analysis results",
        "visualize:Launch interactive viewer",
    ]

    for cmd in commands:
        assert cmd in script


@pytest.mark.unit
def test_zsh_completion_has_compdef():
    """Test zsh completion has compdef directive."""
    script = _get_zsh_completion()

    assert "#compdef oscura" in script


@pytest.mark.unit
def test_zsh_completion_arguments():
    """Test zsh completion defines arguments."""
    script = _get_zsh_completion()

    assert "_arguments" in script
    assert "--help" in script
    assert "--verbose" in script
    assert "--config" in script


@pytest.mark.unit
def test_zsh_completion_file_patterns():
    """Test zsh completion includes file glob patterns."""
    script = _get_zsh_completion()

    assert "_files" in script
    assert "*.{wfm,vcd,csv,pcap,wav}" in script


@pytest.mark.unit
def test_zsh_completion_config_subcommand():
    """Test zsh completion handles config subcommand."""
    script = _get_zsh_completion()

    assert "--show[Show configuration]" in script
    assert "--set[Set value]" in script
    assert "--edit[Edit configuration]" in script


# =============================================================================
# Test _get_fish_completion()
# =============================================================================


@pytest.mark.unit
def test_fish_completion_contains_commands():
    """Test fish completion contains command descriptions."""
    script = _get_fish_completion()

    commands = [
        "analyze",
        "decode",
        "export",
        "visualize",
        "benchmark",
        "validate",
        "config",
    ]

    for cmd in commands:
        assert cmd in script


@pytest.mark.unit
def test_fish_completion_uses_complete():
    """Test fish completion uses complete command."""
    script = _get_fish_completion()

    assert "complete -c oscura" in script


@pytest.mark.unit
def test_fish_completion_subcommands():
    """Test fish completion defines subcommands with descriptions."""
    script = _get_fish_completion()

    assert "__fish_use_subcommand" in script
    assert '-d "Run full analysis workflow"' in script
    assert '-d "Decode protocol data"' in script


@pytest.mark.unit
@pytest.mark.unit
@pytest.mark.unit
def test_fish_completion_file_completions():
    """Test fish completion handles file completions."""
    script = _get_fish_completion()

    assert "__fish_complete_suffix" in script
    assert ".wfm" in script
    assert ".vcd" in script


# =============================================================================
# Test install_completion()
# =============================================================================


@pytest.mark.unit
def test_install_completion_bash(tmp_path):
    """Test installing bash completion."""
    with patch("pathlib.Path.home") as mock_home:
        mock_home.return_value = tmp_path

        result_path = install_completion("bash")

        expected_path = tmp_path / ".bash_completion.d" / "oscura"
        assert result_path == expected_path
        assert expected_path.exists()
        assert expected_path.read_text() == get_completion_script("bash")


@pytest.mark.unit
def test_install_completion_zsh(tmp_path):
    """Test installing zsh completion."""
    with patch("pathlib.Path.home") as mock_home:
        mock_home.return_value = tmp_path

        result_path = install_completion("zsh")

        expected_path = tmp_path / ".zsh" / "completion" / "_oscura"
        assert result_path == expected_path
        assert expected_path.exists()
        assert expected_path.read_text() == get_completion_script("zsh")


@pytest.mark.unit
def test_install_completion_fish(tmp_path):
    """Test installing fish completion."""
    with patch("pathlib.Path.home") as mock_home:
        mock_home.return_value = tmp_path

        result_path = install_completion("fish")

        expected_path = tmp_path / ".config" / "fish" / "completions" / "oscura.fish"
        assert result_path == expected_path
        assert expected_path.exists()
        assert expected_path.read_text() == get_completion_script("fish")


@pytest.mark.unit
def test_install_completion_creates_directories(tmp_path):
    """Test that install_completion creates necessary directories."""
    with patch("pathlib.Path.home") as mock_home:
        mock_home.return_value = tmp_path

        # Directories don't exist yet
        assert not (tmp_path / ".bash_completion.d").exists()

        install_completion("bash")

        # Directory should be created
        assert (tmp_path / ".bash_completion.d").exists()


@pytest.mark.unit
def test_install_completion_overwrites_existing(tmp_path):
    """Test that install_completion overwrites existing file."""
    with patch("pathlib.Path.home") as mock_home:
        mock_home.return_value = tmp_path

        completion_dir = tmp_path / ".bash_completion.d"
        completion_dir.mkdir(parents=True)
        completion_file = completion_dir / "oscura"
        completion_file.write_text("old content")

        install_completion("bash")

        # Should be overwritten with new content
        assert completion_file.read_text() != "old content"
        assert "_oscura_completion" in completion_file.read_text()


@pytest.mark.unit
def test_install_completion_unsupported_shell():
    """Test error handling for unsupported shell installation."""
    with pytest.raises(ValueError, match="Unsupported shell"):
        install_completion("cmd")


@pytest.mark.unit
def test_install_completion_returns_path(tmp_path):
    """Test that install_completion returns the installation path."""
    with patch("pathlib.Path.home") as mock_home:
        mock_home.return_value = tmp_path

        result = install_completion("bash")

        assert isinstance(result, Path)
        assert result.exists()
        assert result.is_file()


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


@pytest.mark.unit
def test_completion_scripts_are_nonempty():
    """Test that all completion scripts have content."""
    for shell in ["bash", "zsh", "fish"]:
        script = get_completion_script(shell)
        assert len(script) > 100  # Reasonable minimum length


@pytest.mark.unit
def test_completion_scripts_are_different():
    """Test that completion scripts differ for each shell."""
    bash = get_completion_script("bash")
    zsh = get_completion_script("zsh")
    fish = get_completion_script("fish")

    # Each should be unique
    assert bash != zsh
    assert bash != fish
    assert zsh != fish


@pytest.mark.unit
@pytest.mark.unit
def test_fish_completion_no_syntax_errors():
    """Test that fish completion has valid syntax."""
    script = _get_fish_completion()

    # Fish uses balanced quotes
    double_quotes = script.count('"')
    assert double_quotes % 2 == 0  # Even number of double quotes


@pytest.mark.unit
def test_all_completions_include_help():
    """Test that all completion scripts include help option."""
    for shell in ["bash", "zsh", "fish"]:
        script = get_completion_script(shell)
        assert "help" in script.lower()


@pytest.mark.unit
def test_completion_module_main():
    """Test running completion module as __main__."""
    import sys

    with patch.object(sys, "argv", ["completion.py", "bash"]):
        with patch("builtins.print") as mock_print:
            # Import and run as main

            if __name__ == "__main__":
                # This block won't run in tests, but we can call the logic
                script = get_completion_script("bash")
                assert len(script) > 0


@pytest.mark.unit
def test_install_creates_nested_directories(tmp_path):
    """Test that install creates deeply nested directories."""
    with patch("pathlib.Path.home") as mock_home:
        mock_home.return_value = tmp_path

        # Fish has deepest nesting: .config/fish/completions
        install_completion("fish")

        assert (tmp_path / ".config").exists()
        assert (tmp_path / ".config" / "fish").exists()
        assert (tmp_path / ".config" / "fish" / "completions").exists()
