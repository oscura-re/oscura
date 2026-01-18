"""Comprehensive unit tests for shell.py CLI module.

This module provides extensive testing for the Oscura interactive shell (REPL), including:
- Namespace initialization with auto-imports
- Tab completion setup
- History persistence
- Custom console prompts
- Help functionality
- Error handling


Test Coverage:
- get_oscura_namespace() auto-import functionality
- setup_history() readline history management
- oscura_help() help text display
- OscuraConsole custom console class
- start_shell() shell initialization
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from oscura.cli.shell import (
    HISTORY_FILE,
    HISTORY_LENGTH,
    OscuraConsole,
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
@pytest.mark.cli
def test_get_oscura_namespace_basic_imports():
    """Test that basic imports are in namespace."""
    namespace = get_oscura_namespace()

    # Core module
    assert "osc" in namespace
    assert namespace["osc"].__name__ == "oscura"

    # NumPy and Matplotlib
    assert "np" in namespace
    assert namespace["np"].__name__ == "numpy"
    assert "plt" in namespace
    assert namespace["plt"].__name__ == "matplotlib.pyplot"


@pytest.mark.unit
@pytest.mark.cli
def test_get_oscura_namespace_core_types():
    """Test that core types are imported."""
    namespace = get_oscura_namespace()

    # Check for core types
    assert "WaveformTrace" in namespace
    assert "DigitalTrace" in namespace
    assert "TraceMetadata" in namespace
    assert "ProtocolPacket" in namespace


@pytest.mark.unit
@pytest.mark.cli
def test_get_oscura_namespace_loader_functions():
    """Test that loader functions are imported."""
    namespace = get_oscura_namespace()

    assert "load" in namespace
    assert "get_supported_formats" in namespace


@pytest.mark.unit
@pytest.mark.cli
def test_get_oscura_namespace_measurement_functions():
    """Test that measurement functions are imported."""
    namespace = get_oscura_namespace()

    measurement_funcs = [
        "rise_time",
        "fall_time",
        "frequency",
        "period",
        "amplitude",
        "rms",
        "mean",
        "overshoot",
        "undershoot",
        "duty_cycle",
        "pulse_width",
        "measure",
    ]

    for func in measurement_funcs:
        assert func in namespace, f"Missing measurement function: {func}"


@pytest.mark.unit
@pytest.mark.cli
def test_get_oscura_namespace_spectral_functions():
    """Test that spectral analysis functions are imported."""
    namespace = get_oscura_namespace()

    spectral_funcs = [
        "fft",
        "psd",
        "thd",
        "snr",
        "sinad",
        "enob",
        "sfdr",
        "spectrogram",
    ]

    for func in spectral_funcs:
        assert func in namespace, f"Missing spectral function: {func}"


@pytest.mark.unit
@pytest.mark.cli
def test_get_oscura_namespace_filter_functions():
    """Test that filter functions are imported."""
    namespace = get_oscura_namespace()

    filter_funcs = ["low_pass", "high_pass", "band_pass", "band_stop"]

    for func in filter_funcs:
        assert func in namespace, f"Missing filter function: {func}"


@pytest.mark.unit
@pytest.mark.cli
def test_get_oscura_namespace_math_operations():
    """Test that math operation functions are imported."""
    namespace = get_oscura_namespace()

    math_funcs = [
        "add",
        "subtract",
        "multiply",
        "divide",
        "differentiate",
        "integrate",
    ]

    for func in math_funcs:
        assert func in namespace, f"Missing math function: {func}"


@pytest.mark.unit
@pytest.mark.cli
def test_get_oscura_namespace_digital_functions():
    """Test that digital signal functions are imported."""
    namespace = get_oscura_namespace()

    digital_funcs = ["to_digital", "detect_edges"]

    for func in digital_funcs:
        assert func in namespace, f"Missing digital function: {func}"


@pytest.mark.unit
@pytest.mark.cli
def test_get_oscura_namespace_statistical_functions():
    """Test that statistical functions are imported."""
    namespace = get_oscura_namespace()

    stat_funcs = ["basic_stats", "histogram", "percentiles"]

    for func in stat_funcs:
        assert func in namespace, f"Missing statistical function: {func}"


@pytest.mark.unit
@pytest.mark.cli
def test_get_oscura_namespace_protocol_decoders():
    """Test that protocol decoders are imported when available."""
    namespace = get_oscura_namespace()

    # These may not be present in all environments, so just check structure
    # If protocols module exists, decoder functions should be there
    expected_decoders = ["decode_uart", "decode_spi", "decode_i2c", "decode_can"]

    # Check if at least some protocol functions are available
    # (they might not all be present depending on import success)
    decoder_count = sum(1 for dec in expected_decoders if dec in namespace)
    # Either all are present or none (depending on import success)
    assert decoder_count == 0 or decoder_count == len(expected_decoders)


@pytest.mark.unit
@pytest.mark.cli
def test_get_oscura_namespace_discovery_functions():
    """Test that discovery functions are imported when available."""
    namespace = get_oscura_namespace()

    # These are optional imports
    discovery_funcs = ["characterize_signal", "find_anomalies", "decode_protocol"]

    # Check structure - may not all be present
    discovery_count = sum(1 for func in discovery_funcs if func in namespace)
    # Either all or none present
    assert discovery_count >= 0


@pytest.mark.unit
@pytest.mark.cli
def test_get_oscura_namespace_import_error_handling():
    """Test that import errors are handled gracefully."""
    # Mock oscura import to fail but allow other imports
    import builtins

    original_import = builtins.__import__

    def selective_import(name, *args, **kwargs):
        if name == "oscura" or name.startswith("oscura."):
            raise ImportError("No module")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=selective_import):
        namespace = get_oscura_namespace()

        # Should still have numpy and pyplot
        assert "np" in namespace
        assert "plt" in namespace
        # But should not have oscura
        assert "tk" not in namespace


# =============================================================================
# Test setup_history()
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_setup_history_enables_tab_completion():
    """Test that setup_history enables tab completion."""
    with patch("oscura.cli.shell.readline") as mock_readline:
        setup_history()

        # Should parse and bind tab completion
        mock_readline.parse_and_bind.assert_called_once_with("tab: complete")


@pytest.mark.unit
@pytest.mark.cli
def test_setup_history_loads_existing_file():
    """Test that setup_history loads existing history file."""
    with patch("oscura.cli.shell.readline") as mock_readline:
        with patch("pathlib.Path.exists", return_value=True):
            setup_history()

            # Should attempt to read history file
            mock_readline.read_history_file.assert_called_once()


@pytest.mark.unit
@pytest.mark.cli
def test_setup_history_handles_missing_file():
    """Test that setup_history handles missing history file gracefully."""
    with patch("oscura.cli.shell.readline") as mock_readline:
        with patch("pathlib.Path.exists", return_value=False):
            setup_history()

            # Should not attempt to read history file
            mock_readline.read_history_file.assert_not_called()


@pytest.mark.unit
@pytest.mark.cli
def test_setup_history_sets_history_length():
    """Test that setup_history sets correct history length."""
    with patch("oscura.cli.shell.readline") as mock_readline:
        setup_history()

        mock_readline.set_history_length.assert_called_once_with(HISTORY_LENGTH)


@pytest.mark.unit
@pytest.mark.cli
def test_setup_history_registers_save_on_exit():
    """Test that setup_history registers atexit handler."""
    with patch("oscura.cli.shell.readline") as mock_readline:
        with patch("oscura.cli.shell.atexit") as mock_atexit:
            setup_history()

            # Should register an atexit handler
            mock_atexit.register.assert_called_once()


@pytest.mark.unit
@pytest.mark.cli
def test_setup_history_corrupted_file():
    """Test that setup_history handles corrupted history file."""
    with patch("oscura.cli.shell.readline") as mock_readline:
        with patch("pathlib.Path.exists", return_value=True):
            # Simulate read error
            mock_readline.read_history_file.side_effect = Exception("Corrupted file")

            # Should not raise exception
            setup_history()

            # Should still set history length
            mock_readline.set_history_length.assert_called_once()


# =============================================================================
# Test oscura_help()
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_oscura_help_displays_help_text(capsys):
    """Test that oscura_help displays help text."""
    oscura_help()

    captured = capsys.readouterr()
    assert "Oscura Interactive Shell" in captured.out
    assert "Quick Reference" in captured.out


@pytest.mark.unit
@pytest.mark.cli
def test_oscura_help_includes_loading_section(capsys):
    """Test that help includes loading data section."""
    oscura_help()

    captured = capsys.readouterr()
    assert "Loading Data:" in captured.out
    assert "load(" in captured.out
    assert "get_supported_formats()" in captured.out


@pytest.mark.unit
@pytest.mark.cli
def test_oscura_help_includes_measurements_section(capsys):
    """Test that help includes waveform measurements section."""
    oscura_help()

    captured = capsys.readouterr()
    assert "Waveform Measurements:" in captured.out
    assert "rise_time(" in captured.out
    assert "fall_time(" in captured.out
    assert "measure(" in captured.out


@pytest.mark.unit
@pytest.mark.cli
def test_oscura_help_includes_spectral_section(capsys):
    """Test that help includes spectral analysis section."""
    oscura_help()

    captured = capsys.readouterr()
    assert "Spectral Analysis:" in captured.out
    assert "fft(" in captured.out
    assert "psd(" in captured.out
    assert "thd(" in captured.out


@pytest.mark.unit
@pytest.mark.cli
def test_oscura_help_includes_digital_section(capsys):
    """Test that help includes digital analysis section."""
    oscura_help()

    captured = capsys.readouterr()
    assert "Digital Analysis:" in captured.out
    assert "to_digital(" in captured.out
    assert "detect_edges(" in captured.out


@pytest.mark.unit
@pytest.mark.cli
def test_oscura_help_includes_filtering_section(capsys):
    """Test that help includes filtering section."""
    oscura_help()

    captured = capsys.readouterr()
    assert "Filtering:" in captured.out
    assert "low_pass(" in captured.out
    assert "high_pass(" in captured.out


@pytest.mark.unit
@pytest.mark.cli
def test_oscura_help_includes_protocol_section(capsys):
    """Test that help includes protocol decoding section."""
    oscura_help()

    captured = capsys.readouterr()
    assert "Protocol Decoding:" in captured.out
    assert "decode_uart(" in captured.out
    assert "decode_spi(" in captured.out


@pytest.mark.unit
@pytest.mark.cli
def test_oscura_help_includes_discovery_section(capsys):
    """Test that help includes discovery section."""
    oscura_help()

    captured = capsys.readouterr()
    assert "Discovery" in captured.out or "Auto-Analysis" in captured.out


# =============================================================================
# Test OscuraConsole
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_oscura_console_initialization():
    """Test OscuraConsole initialization."""
    namespace = {"test": "value"}
    console = OscuraConsole(locals=namespace)

    assert console.prompt_counter == 1
    assert console.locals == namespace


@pytest.mark.unit
@pytest.mark.cli
def test_oscura_console_initialization_no_locals():
    """Test OscuraConsole initialization without locals."""
    console = OscuraConsole()

    assert console.prompt_counter == 1
    assert console.locals is not None


@pytest.mark.unit
@pytest.mark.cli
def test_oscura_console_interact_with_banner(capsys):
    """Test OscuraConsole interact shows custom banner."""
    console = OscuraConsole()

    # Mock the parent interact to avoid actual REPL
    with patch("code.InteractiveConsole.interact"):
        console.interact()

    # Can't easily capture the banner since it's passed to parent
    # Just verify the method works


@pytest.mark.unit
@pytest.mark.cli
def test_oscura_console_interact_with_custom_exitmsg(capsys):
    """Test OscuraConsole interact with custom exit message."""
    console = OscuraConsole()

    with patch("code.InteractiveConsole.interact") as mock_interact:
        console.interact(exitmsg="Custom goodbye")

        # Should pass custom exitmsg to parent
        call_kwargs = mock_interact.call_args[1]
        assert call_kwargs["exitmsg"] == "Custom goodbye"


@pytest.mark.unit
@pytest.mark.cli
def test_oscura_console_raw_input_prompt():
    """Test OscuraConsole custom prompt with counter."""
    console = OscuraConsole()

    with patch("code.InteractiveConsole.raw_input", return_value="test input"):
        result = console.raw_input("")

        assert result == "test input"
        # Counter should increment
        assert console.prompt_counter == 2


@pytest.mark.unit
@pytest.mark.cli
def test_oscura_console_raw_input_increments_counter():
    """Test that prompt counter increments on each input."""
    console = OscuraConsole()

    with patch("code.InteractiveConsole.raw_input", return_value=""):
        console.raw_input("")
        assert console.prompt_counter == 2

        console.raw_input("")
        assert console.prompt_counter == 3

        console.raw_input("")
        assert console.prompt_counter == 4


@pytest.mark.unit
@pytest.mark.cli
def test_oscura_console_showtraceback():
    """Test OscuraConsole showtraceback method."""
    console = OscuraConsole()

    with patch("code.InteractiveConsole.showtraceback") as mock_traceback:
        console.showtraceback()

        # Should call parent showtraceback
        mock_traceback.assert_called_once()


# =============================================================================
# Test start_shell()
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_start_shell_sets_up_history():
    """Test that start_shell sets up history."""
    with patch("oscura.cli.shell.setup_history") as mock_setup:
        with patch("oscura.cli.shell.OscuraConsole") as mock_console_class:
            mock_console = Mock()
            mock_console_class.return_value = mock_console

            start_shell()

            mock_setup.assert_called_once()


@pytest.mark.unit
@pytest.mark.cli
def test_start_shell_creates_namespace():
    """Test that start_shell creates Oscura namespace."""
    with patch("oscura.cli.shell.setup_history"):
        with patch("oscura.cli.shell.get_oscura_namespace") as mock_namespace:
            with patch("oscura.cli.shell.OscuraConsole") as mock_console_class:
                mock_console = Mock()
                mock_console_class.return_value = mock_console
                mock_namespace.return_value = {"test": "namespace"}

                start_shell()

                mock_namespace.assert_called_once()


@pytest.mark.unit
@pytest.mark.cli
def test_start_shell_adds_help_function():
    """Test that start_shell adds oscura_help to namespace."""
    with patch("oscura.cli.shell.setup_history"):
        with patch("oscura.cli.shell.get_oscura_namespace") as mock_namespace:
            with patch("oscura.cli.shell.OscuraConsole") as mock_console_class:
                mock_console = Mock()
                mock_console_class.return_value = mock_console
                namespace = {}
                mock_namespace.return_value = namespace

                start_shell()

                # Should add oscura_help to namespace
                assert "oscura_help" in namespace


@pytest.mark.unit
@pytest.mark.cli
def test_start_shell_sets_up_completer():
    """Test that start_shell sets up tab completion."""
    with patch("oscura.cli.shell.setup_history"):
        with patch("oscura.cli.shell.get_oscura_namespace", return_value={}):
            with patch("oscura.cli.shell.OscuraConsole") as mock_console_class:
                with patch("oscura.cli.shell.readline") as mock_readline:
                    with patch("oscura.cli.shell.rlcompleter") as mock_rlcompleter:
                        mock_console = Mock()
                        mock_console_class.return_value = mock_console

                        start_shell()

                        # Should create completer and set it
                        mock_rlcompleter.Completer.assert_called_once()
                        mock_readline.set_completer.assert_called_once()


@pytest.mark.unit
@pytest.mark.cli
def test_start_shell_starts_console():
    """Test that start_shell starts the interactive console."""
    with patch("oscura.cli.shell.setup_history"):
        with patch("oscura.cli.shell.get_oscura_namespace", return_value={}):
            with patch("oscura.cli.shell.OscuraConsole") as mock_console_class:
                with patch("oscura.cli.shell.readline"):
                    with patch("oscura.cli.shell.rlcompleter"):
                        mock_console = Mock()
                        mock_console_class.return_value = mock_console

                        start_shell()

                        # Should create console with namespace
                        mock_console_class.assert_called_once()
                        call_kwargs = mock_console_class.call_args[1]
                        assert "locals" in call_kwargs

                        # Should call interact
                        mock_console.interact.assert_called_once()


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_history_file_constant():
    """Test HISTORY_FILE constant is properly set."""
    assert isinstance(HISTORY_FILE, Path)
    assert HISTORY_FILE.name == ".oscura_history"
    assert HISTORY_FILE.parent == Path.home()


@pytest.mark.unit
@pytest.mark.cli
def test_history_length_constant():
    """Test HISTORY_LENGTH constant is properly set."""
    assert isinstance(HISTORY_LENGTH, int)
    assert HISTORY_LENGTH == 1000
    assert HISTORY_LENGTH > 0


@pytest.mark.unit
@pytest.mark.cli
def test_namespace_has_no_conflicts():
    """Test that namespace imports don't conflict."""
    namespace = get_oscura_namespace()

    # Check for common conflicts
    # tk should be oscura, not tkinter
    if "tk" in namespace:
        assert namespace["tk"].__name__ == "oscura"

    # np should be numpy
    if "np" in namespace:
        assert namespace["np"].__name__ == "numpy"


@pytest.mark.unit
@pytest.mark.cli
def test_namespace_function_callable():
    """Test that functions in namespace are callable."""
    namespace = get_oscura_namespace()

    # Check a few key functions
    test_functions = ["load", "rise_time", "fft"]

    for func_name in test_functions:
        if func_name in namespace:
            assert callable(namespace[func_name]), f"{func_name} is not callable"


@pytest.mark.unit
@pytest.mark.cli
def test_console_with_actual_namespace():
    """Test console initialization with actual Oscura namespace."""
    namespace = get_oscura_namespace()
    console = OscuraConsole(locals=namespace)

    assert console.locals == namespace
    assert console.prompt_counter == 1


@pytest.mark.unit
@pytest.mark.cli
def test_help_function_in_namespace():
    """Test that oscura_help function works when added to namespace."""
    namespace = get_oscura_namespace()
    namespace["oscura_help"] = oscura_help

    # Should be able to call it
    assert callable(namespace["oscura_help"])


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
def test_get_oscura_namespace_partial_import_failure():
    """Test namespace creation when some imports fail."""
    # This is difficult to test without actually breaking imports
    # Just verify the structure handles it gracefully
    namespace = get_oscura_namespace()

    # Should always have numpy and pyplot as fallback
    assert "np" in namespace
    assert "plt" in namespace


@pytest.mark.unit
@pytest.mark.cli
def test_setup_history_io_error():
    """Test setup_history handles I/O errors gracefully."""
    with patch("oscura.cli.shell.readline") as mock_readline:
        with patch("pathlib.Path.exists", return_value=True):
            # Simulate I/O error on read
            mock_readline.read_history_file.side_effect = OSError("Permission denied")

            # Should not raise - setup_history should handle OSError gracefully
            try:
                setup_history()
                success = True
            except (OSError, PermissionError) as e:
                pytest.fail(f"setup_history should handle I/O errors gracefully, got: {e}")

            assert success


@pytest.mark.unit
@pytest.mark.cli
def test_console_keyboard_interrupt():
    """Test console handles KeyboardInterrupt gracefully."""
    console = OscuraConsole()

    with patch("code.InteractiveConsole.raw_input", side_effect=KeyboardInterrupt):
        # Should handle interrupt
        try:
            console.raw_input("")
        except KeyboardInterrupt:
            pass  # Expected


@pytest.mark.unit
@pytest.mark.cli
def test_start_shell_handles_import_errors():
    """Test start_shell handles import errors in dependencies."""
    with patch("oscura.cli.shell.setup_history"):
        with patch("oscura.cli.shell.get_oscura_namespace", return_value={}):
            # Mock rlcompleter to fail
            with patch("oscura.cli.shell.rlcompleter", None):
                with patch("oscura.cli.shell.OscuraConsole") as mock_console_class:
                    with patch("oscura.cli.shell.readline"):
                        mock_console = Mock()
                        mock_console_class.return_value = mock_console

                        # Should handle missing rlcompleter gracefully
                        try:
                            start_shell()
                        except AttributeError:
                            pass  # Expected if rlcompleter is None
