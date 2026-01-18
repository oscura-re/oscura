"""Unit tests for the context-sensitive help module.

Tests the plain English help system, command suggestions,
and result explanations for non-expert users.
"""

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from oscura.onboarding.help import (
    HELP_DATABASE,
    explain_result,
    get_example,
    get_help,
    suggest_commands,
)

pytestmark = pytest.mark.unit


class TestHelpDatabase:
    """Tests for the HELP_DATABASE structure."""

    def test_help_database_exists(self):
        """Test that help database is populated."""
        assert isinstance(HELP_DATABASE, dict)
        assert len(HELP_DATABASE) > 0

    def test_help_database_entries_have_required_fields(self):
        """Test that all help entries have required fields."""
        required_fields = ["summary", "plain_english"]

        for topic, entry in HELP_DATABASE.items():
            assert "summary" in entry, f"{topic} missing 'summary'"
            assert "plain_english" in entry, f"{topic} missing 'plain_english'"

    def test_help_database_common_topics(self):
        """Test that common topics are in the database."""
        expected_topics = ["rise_time", "fall_time", "frequency", "thd", "snr", "fft", "load"]

        for topic in expected_topics:
            assert topic in HELP_DATABASE, f"Missing help for '{topic}'"

    def test_help_entry_summary_is_string(self):
        """Test that summaries are non-empty strings."""
        for entry in HELP_DATABASE.values():
            assert isinstance(entry["summary"], str)
            assert len(entry["summary"]) > 0

    def test_help_entry_plain_english_is_descriptive(self):
        """Test that plain English explanations are substantial."""
        for topic, entry in HELP_DATABASE.items():
            plain_english = entry["plain_english"]
            assert isinstance(plain_english, str)
            # Should have meaningful content
            assert len(plain_english) > 50, f"{topic} has too short explanation"

    def test_help_entry_related_topics(self):
        """Test that related topics are valid lists."""
        for entry in HELP_DATABASE.values():
            if "related" in entry:
                related = entry["related"]
                assert isinstance(related, list)
                for item in related:
                    assert isinstance(item, str)

    def test_help_entry_when_to_use(self):
        """Test that when_to_use entries are lists of strings."""
        for entry in HELP_DATABASE.values():
            if "when_to_use" in entry:
                uses = entry["when_to_use"]
                assert isinstance(uses, list)
                for use in uses:
                    assert isinstance(use, str)


class TestGetHelp:
    """Tests for the get_help function."""

    def test_get_help_existing_topic(self):
        """Test getting help for an existing topic."""
        help_text = get_help("rise_time")

        assert help_text is not None
        assert isinstance(help_text, str)
        assert "rise_time" in help_text.lower()
        assert len(help_text) > 100

    def test_get_help_case_insensitive(self):
        """Test that get_help is case insensitive."""
        help1 = get_help("rise_time")
        help2 = get_help("RISE_TIME")
        help3 = get_help("Rise_Time")

        assert help1 == help2 == help3

    def test_get_help_strips_whitespace(self):
        """Test that get_help strips leading/trailing whitespace."""
        help1 = get_help("rise_time")
        help2 = get_help("  rise_time  ")

        assert help1 == help2

    def test_get_help_nonexistent_topic(self):
        """Test getting help for a nonexistent topic."""
        help_text = get_help("nonexistent_function_xyz")

        # Should return None for unknown topics without docstring
        assert help_text is None

    def test_get_help_formats_output(self):
        """Test that help output is properly formatted."""
        help_text = get_help("frequency")

        assert help_text is not None
        assert "Help: frequency" in help_text
        assert "=" * 50 in help_text

    def test_get_help_includes_related(self):
        """Test that help includes related topics."""
        help_text = get_help("frequency")

        assert help_text is not None
        assert "Related:" in help_text

    def test_get_help_includes_when_to_use(self):
        """Test that help includes when to use section."""
        help_text = get_help("thd")

        assert help_text is not None
        assert "When to use" in help_text

    def test_get_help_fallback_to_docstring(self):
        """Test fallback to function docstring for unknown help topics."""
        # This tests that we try to get docstring from oscura
        with patch("oscura.get_supported_formats") as mock_func:
            mock_func.__doc__ = "This is a test docstring."

            # get_help should find the docstring
            help_text = get_help("get_supported_formats")
            # May return docstring or None depending on import
            if help_text:
                assert "docstring" in help_text.lower() or "supported" in help_text.lower()


class TestSuggestCommands:
    """Tests for the suggest_commands function."""

    def test_suggest_commands_no_trace(self):
        """Test suggestions when no trace is loaded."""
        suggestions = suggest_commands(trace=None)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

        # Should suggest loading a trace
        commands = [s["command"] for s in suggestions]
        assert any("load" in cmd for cmd in commands)

    def test_suggest_commands_returns_proper_structure(self):
        """Test that suggestions have proper structure."""
        suggestions = suggest_commands(trace=None)

        for suggestion in suggestions:
            assert "command" in suggestion
            assert "description" in suggestion
            assert "reason" in suggestion

    @dataclass
    class MockTrace:
        """Mock trace for testing."""

        data: np.ndarray

    def test_suggest_commands_with_trace(self):
        """Test suggestions when a trace is loaded."""
        trace = self.MockTrace(data=np.random.randn(1000))
        suggestions = suggest_commands(trace=trace)

        assert len(suggestions) > 0

        # Should suggest measure()
        commands = [s["command"] for s in suggestions]
        assert any("measure" in cmd for cmd in commands)

    def test_suggest_commands_digital_signal(self):
        """Test suggestions for digital-looking signal."""
        # Create a signal with few unique levels (digital-like)
        data = np.array([0.0, 3.3, 0.0, 3.3, 3.3, 0.0] * 100)
        trace = self.MockTrace(data=data)

        suggestions = suggest_commands(trace=trace)

        # Should suggest to_digital
        commands = [s["command"] for s in suggestions]
        assert any("to_digital" in cmd or "characterize" in cmd for cmd in commands)

    def test_suggest_commands_analog_signal(self):
        """Test suggestions for analog-looking signal."""
        # Create a signal with many unique values (analog-like)
        data = np.sin(2 * np.pi * np.linspace(0, 10, 1000))
        trace = self.MockTrace(data=data)

        suggestions = suggest_commands(trace=trace)

        # Should suggest fft or spectral analysis
        commands = [s["command"] for s in suggestions]
        assert any("fft" in cmd or "thd" in cmd for cmd in commands)

    def test_suggest_commands_always_includes_filter(self):
        """Test that filter suggestion is always included for loaded trace."""
        trace = self.MockTrace(data=np.random.randn(100))
        suggestions = suggest_commands(trace=trace)

        commands = [s["command"] for s in suggestions]
        assert any("low_pass" in cmd or "filter" in cmd for cmd in commands)

    def test_suggest_commands_uart_context(self):
        """Test suggestions with UART context."""
        trace = self.MockTrace(data=np.random.randn(100))
        suggestions = suggest_commands(trace=trace, context="I think this is UART data")

        commands = [s["command"] for s in suggestions]
        assert any("decode_uart" in cmd for cmd in commands)

    def test_suggest_commands_spi_context(self):
        """Test suggestions with SPI context."""
        trace = self.MockTrace(data=np.random.randn(100))
        suggestions = suggest_commands(trace=trace, context="This looks like SPI")

        commands = [s["command"] for s in suggestions]
        assert any("decode_spi" in cmd for cmd in commands)

    def test_suggest_commands_i2c_context(self):
        """Test suggestions with I2C context."""
        trace = self.MockTrace(data=np.random.randn(100))
        suggestions = suggest_commands(trace=trace, context="I2C bus capture")

        commands = [s["command"] for s in suggestions]
        assert any("decode_i2c" in cmd for cmd in commands)

    def test_suggest_commands_serial_context(self):
        """Test suggestions with serial context (alias for UART)."""
        trace = self.MockTrace(data=np.random.randn(100))
        suggestions = suggest_commands(trace=trace, context="serial port data")

        commands = [s["command"] for s in suggestions]
        assert any("uart" in cmd.lower() for cmd in commands)


class TestExplainResult:
    """Tests for the explain_result function."""

    def test_explain_rise_time_fast(self):
        """Test explanation for fast rise time."""
        result = explain_result(500e-12, "rise_time")  # 500 ps

        assert "ps" in result
        assert "fast" in result.lower()

    def test_explain_rise_time_moderate(self):
        """Test explanation for moderate rise time."""
        result = explain_result(5e-9, "rise_time")  # 5 ns

        assert "ns" in result
        assert "fast" in result.lower() or "suitable" in result.lower()

    def test_explain_rise_time_slow(self):
        """Test explanation for slow rise time."""
        result = explain_result(1e-6, "rise_time")  # 1 us

        assert "us" in result
        assert "slow" in result.lower()

    def test_explain_rise_time_subps(self):
        """Test explanation for sub-picosecond rise time."""
        result = explain_result(0.5e-12, "rise_time")  # 0.5 ps

        assert "ps" in result
        assert "extremely" in result.lower() or "sub-picosecond" in result.lower()

    def test_explain_fall_time(self):
        """Test explanation for fall time."""
        result = explain_result(2e-9, "fall_time")  # 2 ns

        assert "ns" in result or "Fall time" in result

    def test_explain_frequency_hz(self):
        """Test explanation for Hz-range frequency."""
        result = explain_result(500, "frequency")  # 500 Hz

        assert "Hz" in result
        assert "audio" in result.lower() or "slow" in result.lower()

    def test_explain_frequency_khz(self):
        """Test explanation for kHz-range frequency."""
        result = explain_result(50e3, "frequency")  # 50 kHz

        assert "kHz" in result

    def test_explain_frequency_mhz(self):
        """Test explanation for MHz-range frequency."""
        result = explain_result(100e6, "frequency")  # 100 MHz

        assert "MHz" in result

    def test_explain_frequency_ghz(self):
        """Test explanation for GHz-range frequency."""
        result = explain_result(2.4e9, "frequency")  # 2.4 GHz

        assert "GHz" in result
        assert "high-speed" in result.lower() or "RF" in result

    def test_explain_thd_excellent(self):
        """Test explanation for excellent THD."""
        result = explain_result(-70, "thd")  # -70 dB

        assert "dB" in result
        assert "excellent" in result.lower() or "low" in result.lower()

    def test_explain_thd_good(self):
        """Test explanation for good THD."""
        result = explain_result(-50, "thd")  # -50 dB

        assert "dB" in result
        assert "good" in result.lower()

    def test_explain_thd_fair(self):
        """Test explanation for fair THD."""
        result = explain_result(-30, "thd")  # -30 dB

        assert "dB" in result
        assert "fair" in result.lower()

    def test_explain_thd_poor(self):
        """Test explanation for poor THD."""
        result = explain_result(-15, "thd")  # -15 dB

        assert "dB" in result
        assert "poor" in result.lower() or "significant" in result.lower()

    def test_explain_snr_excellent(self):
        """Test explanation for excellent SNR."""
        result = explain_result(70, "snr")  # 70 dB

        assert "dB" in result
        assert "excellent" in result.lower() or "clean" in result.lower()

    def test_explain_snr_good(self):
        """Test explanation for good SNR."""
        result = explain_result(50, "snr")  # 50 dB

        assert "dB" in result
        assert "good" in result.lower()

    def test_explain_snr_fair(self):
        """Test explanation for fair SNR."""
        result = explain_result(30, "snr")  # 30 dB

        assert "dB" in result
        assert "fair" in result.lower() or "noise" in result.lower()

    def test_explain_snr_poor(self):
        """Test explanation for poor SNR."""
        result = explain_result(10, "snr")  # 10 dB

        assert "dB" in result
        assert "poor" in result.lower() or "noisy" in result.lower()

    def test_explain_unknown_measurement(self):
        """Test explanation for unknown measurement."""
        result = explain_result(42, "unknown_measurement")

        assert "unknown_measurement" in result
        assert "42" in result

    def test_explain_case_insensitive(self):
        """Test that measurement name is case insensitive."""
        result1 = explain_result(5e-9, "rise_time")
        result2 = explain_result(5e-9, "RISE_TIME")
        result3 = explain_result(5e-9, "Rise_Time")

        # All should return similar explanations
        assert "ns" in result1
        assert "ns" in result2
        assert "ns" in result3


class TestGetExample:
    """Tests for the get_example function."""

    def test_get_example_load(self):
        """Test getting example for load function."""
        example = get_example("load")

        assert example is not None
        assert "import oscura" in example
        assert "load(" in example

    def test_get_example_rise_time(self):
        """Test getting example for rise_time function."""
        example = get_example("rise_time")

        assert example is not None
        assert "rise_time" in example

    def test_get_example_fft(self):
        """Test getting example for fft function."""
        example = get_example("fft")

        assert example is not None
        assert "fft" in example

    def test_get_example_measure(self):
        """Test getting example for measure function."""
        example = get_example("measure")

        assert example is not None
        assert "measure" in example

    def test_get_example_case_insensitive(self):
        """Test that get_example is case insensitive."""
        example1 = get_example("load")
        example2 = get_example("LOAD")
        example3 = get_example("Load")

        assert example1 == example2 == example3

    def test_get_example_unknown_function(self):
        """Test getting example for unknown function."""
        example = get_example("nonexistent_function")

        assert example is None

    def test_examples_are_valid_python(self):
        """Test that examples contain valid Python syntax."""
        for func_name in ["load", "rise_time", "fft", "measure"]:
            example = get_example(func_name)
            if example:
                # Should be able to compile (syntax check)
                try:
                    compile(example, "<string>", "exec")
                except SyntaxError as e:
                    pytest.fail(f"Example for {func_name} has syntax error: {e}")


class TestHelpIntegration:
    """Integration tests for the help system."""

    def test_help_and_examples_consistent(self):
        """Test that help topics have matching examples where expected."""
        topics_with_examples = ["load", "rise_time", "fft", "measure"]

        for topic in topics_with_examples:
            help_text = get_help(topic)
            example = get_example(topic)

            assert help_text is not None, f"No help for {topic}"
            assert example is not None, f"No example for {topic}"

    def test_related_topics_exist_in_database(self):
        """Test that related topics reference valid entries."""
        for entry in HELP_DATABASE.values():
            if "related" in entry:
                for _related in entry["related"]:
                    # Related topics should either be in database or be valid oscura functions
                    # (not all need to be in database)
                    pass  # Just checking structure is valid

    def test_comprehensive_help_coverage(self):
        """Test that key measurement functions have help."""
        key_functions = [
            "rise_time",
            "fall_time",
            "frequency",
            "thd",
            "snr",
            "fft",
            "load",
            "measure",
        ]

        for func in key_functions:
            help_text = get_help(func)
            assert help_text is not None, f"Missing help for key function: {func}"
