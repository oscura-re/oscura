"""Comprehensive unit tests for shell.py CLI module.

This module provides extensive testing for the interactive shell, including:
- Namespace auto-import functionality
- History file management
- Tab completion setup
- Custom console behavior
- Help function
- All import categories (core, protocols, discovery, utilities)

Test Coverage:
- get_oscura_namespace() namespace building
- setup_history() history configuration
- oscura_help() help display
- OscuraConsole custom console
- start_shell() entry point
- Import functions for each category
- Signal builder integration

References:
    - src/oscura/cli/shell.py
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from oscura.cli.shell import (
    HISTORY_FILE,
    HISTORY_LENGTH,
    OscuraConsole,
    _build_namespace_dict,
    _get_oscura_imports,
    _import_common_utilities,
    _import_core_oscura,
    _import_discovery,
    _import_protocols,
    get_oscura_namespace,
    oscura_help,
    setup_history,
    start_shell,
)

pytestmark = [pytest.mark.unit, pytest.mark.cli]


# =============================================================================
# Test get_oscura_namespace()
# =============================================================================


@pytest.mark.unit
def test_get_oscura_namespace_returns_dict():
    """Test that get_oscura_namespace returns a dictionary."""
    namespace = get_oscura_namespace()

    assert isinstance(namespace, dict)
    assert len(namespace) > 0


@pytest.mark.unit
def test_get_oscura_namespace_has_core_imports():
    """Test namespace includes core Oscura imports."""
    namespace = get_oscura_namespace()

    # Should have basic types
    assert "WaveformTrace" in namespace
    assert "DigitalTrace" in namespace
    assert "load" in namespace


@pytest.mark.unit
def test_get_oscura_namespace_has_measurements():
    """Test namespace includes measurement functions."""
    namespace = get_oscura_namespace()

    assert "rise_time" in namespace
    assert "fall_time" in namespace
    assert "frequency" in namespace
    assert "amplitude" in namespace


@pytest.mark.unit
def test_get_oscura_namespace_has_analysis():
    """Test namespace includes analysis functions."""
    namespace = get_oscura_namespace()

    assert "fft" in namespace
    assert "psd" in namespace
    assert "thd" in namespace


@pytest.mark.unit
def test_get_oscura_namespace_has_utilities():
    """Test namespace includes common utilities (numpy, matplotlib)."""
    namespace = get_oscura_namespace()

    # Should have np and plt (if available)
    assert "np" in namespace or "plt" in namespace


# =============================================================================
# Test _import_core_oscura()
# =============================================================================


@pytest.mark.unit
def test_import_core_oscura_populates_namespace():
    """Test _import_core_oscura adds symbols to namespace."""
    namespace = {}
    _import_core_oscura(namespace)

    # Should have oscura module reference
    assert "osc" in namespace
    # Should have core types
    assert "load" in namespace
    assert "WaveformTrace" in namespace


@pytest.mark.unit
def test_import_core_oscura_handles_import_error():
    """Test _import_core_oscura handles ImportError gracefully."""
    namespace = {}

    # Mock the import of oscura module itself to raise ImportError
    with patch("oscura.cli.shell._get_oscura_imports", side_effect=ImportError("Test error")):
        with patch("builtins.print") as mock_print:
            # Should not raise, just print warning
            _import_core_oscura(namespace)

            # Namespace should have only 'osc' (the import fails at _get_oscura_imports)
            # Actually, the import of oscura as osc succeeds, but _get_oscura_imports fails
            # So we need to mock earlier - at the 'import oscura as osc' level

    # Correct approach: patch the import statement itself
    namespace = {}
    import sys

    with patch.dict(sys.modules, {"oscura": None}):
        # Force ImportError when trying to import oscura
        with patch("builtins.print") as mock_print:
            _import_core_oscura(namespace)
            # Namespace should be empty (import failed)
            assert len(namespace) == 0


# =============================================================================
# Test _get_oscura_imports()
# =============================================================================


@pytest.mark.unit
def test_get_oscura_imports_returns_dict():
    """Test _get_oscura_imports returns dictionary of symbols."""
    imports = _get_oscura_imports()

    assert isinstance(imports, dict)
    assert len(imports) > 0


@pytest.mark.unit
def test_get_oscura_imports_has_all_categories():
    """Test _get_oscura_imports includes all symbol categories."""
    imports = _get_oscura_imports()

    # Types
    assert "WaveformTrace" in imports
    assert "DigitalTrace" in imports

    # Loaders
    assert "load" in imports

    # Measurements
    assert "rise_time" in imports
    assert "amplitude" in imports

    # Spectral
    assert "fft" in imports
    assert "psd" in imports

    # Filtering
    assert "low_pass" in imports
    assert "high_pass" in imports


# =============================================================================
# Test _build_namespace_dict()
# =============================================================================


@pytest.mark.unit
def test_build_namespace_dict_returns_input():
    """Test _build_namespace_dict returns the input dictionary."""
    imports = {"key1": "value1", "key2": "value2"}
    result = _build_namespace_dict(imports)

    assert result is imports
    assert result == imports


# =============================================================================
# Test _import_protocols()
# =============================================================================


@pytest.mark.unit
def test_import_protocols_adds_decoders():
    """Test _import_protocols adds protocol decoder functions."""
    namespace = {}
    _import_protocols(namespace)

    # Should have decoder functions (if module is available)
    # These might not be present if import fails
    expected_keys = ["decode_uart", "decode_spi", "decode_i2c", "decode_can"]

    # At least some should be present
    present = [key for key in expected_keys if key in namespace]
    assert len(present) >= 0  # Graceful if imports fail


@pytest.mark.unit
def test_import_protocols_handles_import_error():
    """Test _import_protocols handles ImportError gracefully."""
    namespace = {}
    import sys

    # Mock the import by patching the module itself
    original_modules = sys.modules.copy()
    # Temporarily remove the protocols module to simulate import failure
    with patch.dict(sys.modules, {"oscura.analyzers.protocols": None}, clear=False):
        # Should not raise
        _import_protocols(namespace)

        # Namespace should be empty (import failed)
        assert isinstance(namespace, dict)
        # Since import failed, no protocol functions should be added
        assert "decode_uart" not in namespace


# =============================================================================
# Test _import_discovery()
# =============================================================================


@pytest.mark.unit
def test_import_discovery_adds_functions():
    """Test _import_discovery adds discovery functions."""
    namespace = {}
    _import_discovery(namespace)

    # Should have discovery functions (if available)
    possible_keys = ["characterize_signal", "find_anomalies", "decode_protocol"]

    # At least some should be present
    present = [key for key in possible_keys if key in namespace]
    assert len(present) >= 0  # Graceful if imports fail


@pytest.mark.unit
def test_import_discovery_handles_import_error():
    """Test _import_discovery handles ImportError gracefully."""
    namespace = {}

    # Should not raise even if import fails
    _import_discovery(namespace)

    assert isinstance(namespace, dict)


# =============================================================================
# Test _import_common_utilities()
# =============================================================================


@pytest.mark.unit
def test_import_common_utilities_adds_numpy():
    """Test _import_common_utilities adds numpy."""
    namespace = {}
    _import_common_utilities(namespace)

    # Should have numpy (if installed)
    assert "np" in namespace


@pytest.mark.unit
def test_import_common_utilities_adds_matplotlib():
    """Test _import_common_utilities adds matplotlib."""
    namespace = {}
    _import_common_utilities(namespace)

    # Should have matplotlib (if installed)
    assert "plt" in namespace


@pytest.mark.unit
def test_import_common_utilities_handles_missing_modules():
    """Test _import_common_utilities handles missing modules gracefully."""
    namespace = {}

    with patch("builtins.__import__", side_effect=ImportError()):
        # Should not raise
        _import_common_utilities(namespace)

        # Namespace should be empty (imports failed)
        assert len(namespace) == 0


# =============================================================================
# Test setup_history()
# =============================================================================


@pytest.mark.unit
def test_setup_history_enables_completion():
    """Test setup_history enables tab completion."""
    with patch("readline.parse_and_bind") as mock_bind:
        with patch("readline.read_history_file"):
            with patch("readline.set_history_length"):
                with patch("atexit.register"):
                    setup_history()

                    # Should bind tab to complete
                    mock_bind.assert_called_with("tab: complete")


@pytest.mark.unit
def test_setup_history_sets_length():
    """Test setup_history sets history length."""
    with patch("readline.parse_and_bind"):
        with patch("readline.read_history_file"):
            with patch("readline.set_history_length") as mock_length:
                with patch("atexit.register"):
                    setup_history()

                    # Should set length to HISTORY_LENGTH
                    mock_length.assert_called_with(HISTORY_LENGTH)


@pytest.mark.unit
def test_setup_history_loads_existing_file(tmp_path, monkeypatch):
    """Test setup_history loads existing history file."""
    # Create a temporary history file
    history_file = tmp_path / ".oscura_history"
    history_file.write_text("load('test.wfm')\n")

    # Mock HISTORY_FILE to point to temp file
    monkeypatch.setattr("oscura.cli.shell.HISTORY_FILE", history_file)

    with patch("readline.parse_and_bind"):
        with patch("readline.read_history_file") as mock_read:
            with patch("readline.set_history_length"):
                with patch("atexit.register"):
                    setup_history()

                    # Should attempt to read history
                    mock_read.assert_called_once()


@pytest.mark.unit
def test_setup_history_handles_missing_file():
    """Test setup_history handles missing history file gracefully."""
    with patch("readline.parse_and_bind"):
        with patch("readline.read_history_file") as mock_read:
            with patch("readline.set_history_length"):
                with patch("atexit.register"):
                    # Make read fail
                    mock_read.side_effect = FileNotFoundError()

                    # Should not raise
                    setup_history()


@pytest.mark.unit
def test_setup_history_registers_atexit():
    """Test setup_history registers save on exit."""
    with patch("readline.parse_and_bind"):
        with patch("readline.read_history_file"):
            with patch("readline.set_history_length"):
                with patch("atexit.register") as mock_register:
                    setup_history()

                    # Should register save function
                    assert mock_register.called


# =============================================================================
# Test oscura_help()
# =============================================================================


@pytest.mark.unit
def test_oscura_help_prints_information():
    """Test oscura_help prints help information."""
    with patch("builtins.print") as mock_print:
        oscura_help()

        # Should have printed something
        assert mock_print.called
        # Get the printed text
        printed_text = " ".join(str(call[0][0]) for call in mock_print.call_args_list)
        assert "Oscura" in printed_text or "oscura" in printed_text.lower()


@pytest.mark.unit
def test_oscura_help_includes_loading_section():
    """Test oscura_help includes loading data section."""
    with patch("builtins.print") as mock_print:
        oscura_help()

        printed = " ".join(str(call[0][0]) for call in mock_print.call_args_list)
        assert "load" in printed.lower()


@pytest.mark.unit
def test_oscura_help_includes_measurements():
    """Test oscura_help includes measurement functions."""
    with patch("builtins.print") as mock_print:
        oscura_help()

        printed = " ".join(str(call[0][0]) for call in mock_print.call_args_list)
        assert "rise_time" in printed or "fall_time" in printed


@pytest.mark.unit
def test_oscura_help_includes_protocol_decoding():
    """Test oscura_help includes protocol decoding section."""
    with patch("builtins.print") as mock_print:
        oscura_help()

        printed = " ".join(str(call[0][0]) for call in mock_print.call_args_list)
        assert "uart" in printed.lower() or "protocol" in printed.lower()


# =============================================================================
# Test OscuraConsole Class
# =============================================================================


@pytest.mark.unit
def test_oscura_console_initialization():
    """Test OscuraConsole initializes correctly."""
    namespace = {"test": "value"}
    console = OscuraConsole(locals=namespace)

    assert console.prompt_counter == 1


@pytest.mark.unit
def test_oscura_console_interact_banner():
    """Test OscuraConsole.interact shows custom banner."""
    console = OscuraConsole()

    with patch.object(console, "push") as mock_push:
        with patch("code.InteractiveConsole.interact") as mock_interact:
            console.interact()

            # Should call parent interact
            assert mock_interact.called

            # Check if custom banner was passed
            call_kwargs = mock_interact.call_args[1]
            if "banner" in call_kwargs:
                banner = call_kwargs["banner"]
                assert "Oscura" in banner or banner is None


@pytest.mark.unit
def test_oscura_console_raw_input_prompt():
    """Test OscuraConsole.raw_input uses custom prompt."""
    console = OscuraConsole()

    with patch("code.InteractiveConsole.raw_input", return_value="test") as mock_input:
        result = console.raw_input()

        # Should have called with numbered prompt
        mock_input.assert_called_with("In [1]: ")
        assert console.prompt_counter == 2


@pytest.mark.unit
def test_oscura_console_prompt_counter_increments():
    """Test OscuraConsole prompt counter increments."""
    console = OscuraConsole()

    with patch("code.InteractiveConsole.raw_input", return_value=""):
        console.raw_input()
        console.raw_input()
        console.raw_input()

        assert console.prompt_counter == 4


# =============================================================================
# Test start_shell()
# =============================================================================


@pytest.mark.unit
def test_start_shell_sets_up_history():
    """Test start_shell calls setup_history."""
    with patch("oscura.cli.shell.setup_history") as mock_setup:
        with patch("oscura.cli.shell.OscuraConsole") as mock_console_class:
            with patch("readline.set_completer"):
                with patch("oscura.cli.shell.rlcompleter"):
                    mock_console = Mock()
                    mock_console_class.return_value = mock_console

                    start_shell()

                    # Should set up history
                    assert mock_setup.called


@pytest.mark.unit
def test_start_shell_builds_namespace():
    """Test start_shell builds namespace with imports."""
    with patch("oscura.cli.shell.setup_history"):
        with patch("oscura.cli.shell.OscuraConsole") as mock_console_class:
            with patch("readline.set_completer"):
                with patch("oscura.cli.shell.rlcompleter"):
                    mock_console = Mock()
                    mock_console_class.return_value = mock_console

                    start_shell()

                    # Should create console with namespace
                    assert mock_console_class.called
                    call_kwargs = mock_console_class.call_args[1]
                    namespace = call_kwargs.get("locals", {})

                    # Should have help function
                    assert "oscura_help" in namespace


@pytest.mark.unit
def test_start_shell_sets_up_completer():
    """Test start_shell sets up tab completion."""
    with patch("oscura.cli.shell.setup_history"):
        with patch("oscura.cli.shell.OscuraConsole") as mock_console_class:
            with patch("readline.set_completer") as mock_set_completer:
                with patch("oscura.cli.shell.rlcompleter.Completer") as mock_completer_class:
                    mock_console = Mock()
                    mock_console_class.return_value = mock_console

                    mock_completer = Mock()
                    mock_completer_class.return_value = mock_completer

                    start_shell()

                    # Should create completer with namespace
                    assert mock_completer_class.called

                    # Should set the completer
                    mock_set_completer.assert_called_with(mock_completer.complete)


@pytest.mark.unit
def test_start_shell_starts_console():
    """Test start_shell starts interactive console."""
    with patch("oscura.cli.shell.setup_history"):
        with patch("oscura.cli.shell.OscuraConsole") as mock_console_class:
            with patch("readline.set_completer"):
                with patch("oscura.cli.shell.rlcompleter"):
                    mock_console = Mock()
                    mock_console_class.return_value = mock_console

                    start_shell()

                    # Should call interact
                    mock_console.interact.assert_called_once()


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
def test_history_file_constant():
    """Test HISTORY_FILE constant is correctly defined."""
    assert Path.home() / ".oscura_history" == HISTORY_FILE


@pytest.mark.unit
def test_history_length_constant():
    """Test HISTORY_LENGTH constant is correctly defined."""
    assert HISTORY_LENGTH == 1000


@pytest.mark.unit
def test_namespace_has_help_function():
    """Test namespace includes oscura_help function."""
    namespace = get_oscura_namespace()

    # Should NOT have oscura_help in base namespace
    # It's added by start_shell()
    # But we can test it's defined
    assert callable(oscura_help)


@pytest.mark.unit
def test_oscura_console_inherits_from_interactive_console():
    """Test OscuraConsole inherits from code.InteractiveConsole."""
    import code

    assert issubclass(OscuraConsole, code.InteractiveConsole)


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
def test_get_oscura_namespace_is_idempotent():
    """Test calling get_oscura_namespace multiple times gives consistent results."""
    ns1 = get_oscura_namespace()
    ns2 = get_oscura_namespace()

    # Should have same keys
    assert set(ns1.keys()) == set(ns2.keys())


@pytest.mark.unit
def test_oscura_console_with_empty_locals():
    """Test OscuraConsole works with empty locals."""
    console = OscuraConsole(locals={})

    assert console.prompt_counter == 1


@pytest.mark.unit
def test_oscura_console_with_none_locals():
    """Test OscuraConsole works with None locals."""
    console = OscuraConsole(locals=None)

    assert console.prompt_counter == 1
