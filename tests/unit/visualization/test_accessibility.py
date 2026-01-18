"""Unit tests for visualization accessibility features.

Tests:
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from oscura.visualization.accessibility import (
    FAIL_SYMBOL,
    LINE_STYLES,
    PASS_SYMBOL,
    KeyboardHandler,
    add_plot_aria_attributes,
    format_pass_fail,
    generate_alt_text,
    get_colorblind_palette,
    get_multi_line_styles,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

pytestmark = [pytest.mark.unit, pytest.mark.visualization]


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_data() -> np.ndarray:
    """Create sample signal data for testing."""
    return np.array([1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0])


@pytest.fixture
def sample_stats() -> dict[str, float]:
    """Create sample statistics dictionary."""
    return {
        "min": -2.0,
        "max": 3.0,
        "mean": 0.5,
        "std": 1.5,
        "n_samples": 10,
    }


@pytest.fixture
def fig_and_ax() -> tuple[Figure, plt.Axes]:
    """Create a matplotlib figure and axes for testing."""
    fig, ax = plt.subplots()
    yield fig, ax
    plt.close(fig)


# ============================================================================
# Tests for get_colorblind_palette (ACC-001)
# ============================================================================


@pytest.mark.unit
@pytest.mark.visualization
class TestGetColorblindPalette:
    """Tests for get_colorblind_palette function."""

    def test_default_viridis(self) -> None:
        """Test default colormap returns viridis."""
        result = get_colorblind_palette()
        assert result == "viridis"

    def test_viridis_explicit(self) -> None:
        """Test explicit viridis selection."""
        result = get_colorblind_palette("viridis")
        assert result == "viridis"

    def test_cividis(self) -> None:
        """Test cividis colormap selection."""
        result = get_colorblind_palette("cividis")
        assert result == "cividis"

    def test_plasma(self) -> None:
        """Test plasma colormap selection."""
        result = get_colorblind_palette("plasma")
        assert result == "plasma"

    def test_inferno(self) -> None:
        """Test inferno colormap selection."""
        result = get_colorblind_palette("inferno")
        assert result == "inferno"

    def test_magma(self) -> None:
        """Test magma colormap selection."""
        result = get_colorblind_palette("magma")
        assert result == "magma"

    def test_all_valid_names(self) -> None:
        """Test all valid colormap names."""
        valid_names = ["viridis", "cividis", "plasma", "inferno", "magma"]
        for name in valid_names:
            result = get_colorblind_palette(name)
            assert result == name

    def test_invalid_name_raises_error(self) -> None:
        """Test that invalid colormap name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown colormap: jet"):
            get_colorblind_palette("jet")  # type: ignore[arg-type]

    def test_invalid_name_suggests_valid_options(self) -> None:
        """Test that error message includes valid options."""
        with pytest.raises(
            ValueError, match="Valid options: viridis, cividis, plasma, inferno, magma"
        ):
            get_colorblind_palette("rainbow")  # type: ignore[arg-type]

    def test_case_sensitive(self) -> None:
        """Test that colormap names are case-sensitive."""
        with pytest.raises(ValueError, match="Unknown colormap: Viridis"):
            get_colorblind_palette("Viridis")  # type: ignore[arg-type]


# ============================================================================
# Tests for get_multi_line_styles (ACC-001)
# ============================================================================


@pytest.mark.unit
@pytest.mark.visualization
class TestGetMultiLineStyles:
    """Tests for get_multi_line_styles function."""

    def test_single_line(self) -> None:
        """Test generating style for single line."""
        styles = get_multi_line_styles(1)
        assert len(styles) == 1
        color, linestyle = styles[0]
        assert len(color) == 4  # RGBA tuple
        assert linestyle in LINE_STYLES

    def test_multiple_lines(self) -> None:
        """Test generating styles for multiple lines."""
        n_lines = 5
        styles = get_multi_line_styles(n_lines)
        assert len(styles) == n_lines

    def test_returns_rgba_tuples(self) -> None:
        """Test that colors are RGBA tuples."""
        styles = get_multi_line_styles(3)
        for color, _ in styles:
            assert len(color) == 4
            # Check all values are in [0, 1] range
            assert all(0 <= c <= 1 for c in color)

    def test_returns_valid_linestyles(self) -> None:
        """Test that all linestyles are valid."""
        styles = get_multi_line_styles(10)
        for _, linestyle in styles:
            assert linestyle in LINE_STYLES

    def test_linestyle_cycling(self) -> None:
        """Test that line styles cycle when n_lines > len(LINE_STYLES)."""
        n_lines = len(LINE_STYLES) * 2 + 1
        styles = get_multi_line_styles(n_lines)

        # First line style should match line at len(LINE_STYLES)
        _, first_ls = styles[0]
        _, cycled_ls = styles[len(LINE_STYLES)]
        assert first_ls == cycled_ls

    def test_different_colors(self) -> None:
        """Test that different lines get different colors."""
        styles = get_multi_line_styles(4)
        colors = [color for color, _ in styles]

        # All colors should be different
        for i, color1 in enumerate(colors):
            for _j, color2 in enumerate(colors[i + 1 :], start=i + 1):
                assert color1 != color2

    def test_zero_lines(self) -> None:
        """Test with zero lines."""
        styles = get_multi_line_styles(0)
        assert len(styles) == 0

    def test_large_number_of_lines(self) -> None:
        """Test with large number of lines."""
        styles = get_multi_line_styles(100)
        assert len(styles) == 100


# ============================================================================
# Tests for format_pass_fail (ACC-001)
# ============================================================================


@pytest.mark.unit
@pytest.mark.visualization
class TestFormatPassFail:
    """Tests for format_pass_fail function."""

    def test_pass_with_color_and_symbols(self) -> None:
        """Test pass formatting with color and symbols."""
        result = format_pass_fail(True, use_color=True, use_symbols=True)
        assert PASS_SYMBOL in result
        assert "PASS" in result
        assert "\033[92m" in result  # Green color code
        assert "\033[0m" in result  # Reset code

    def test_fail_with_color_and_symbols(self) -> None:
        """Test fail formatting with color and symbols."""
        result = format_pass_fail(False, use_color=True, use_symbols=True)
        assert FAIL_SYMBOL in result
        assert "FAIL" in result
        assert "\033[91m" in result  # Red color code
        assert "\033[0m" in result  # Reset code

    def test_pass_no_color(self) -> None:
        """Test pass formatting without color."""
        result = format_pass_fail(True, use_color=False, use_symbols=True)
        assert PASS_SYMBOL in result
        assert "PASS" in result
        assert "\033[" not in result  # No color codes

    def test_fail_no_color(self) -> None:
        """Test fail formatting without color."""
        result = format_pass_fail(False, use_color=False, use_symbols=True)
        assert FAIL_SYMBOL in result
        assert "FAIL" in result
        assert "\033[" not in result  # No color codes

    def test_pass_no_symbols(self) -> None:
        """Test pass formatting without symbols."""
        result = format_pass_fail(True, use_color=True, use_symbols=False)
        assert PASS_SYMBOL not in result
        assert "PASS" in result
        assert "\033[92m" in result  # Green color code

    def test_fail_no_symbols(self) -> None:
        """Test fail formatting without symbols."""
        result = format_pass_fail(False, use_color=True, use_symbols=False)
        assert FAIL_SYMBOL not in result
        assert "FAIL" in result
        assert "\033[91m" in result  # Red color code

    def test_pass_minimal(self) -> None:
        """Test pass formatting with no color or symbols."""
        result = format_pass_fail(True, use_color=False, use_symbols=False)
        assert result == "PASS"

    def test_fail_minimal(self) -> None:
        """Test fail formatting with no color or symbols."""
        result = format_pass_fail(False, use_color=False, use_symbols=False)
        assert result == "FAIL"

    def test_default_args_pass(self) -> None:
        """Test default arguments for pass (color and symbols enabled)."""
        result = format_pass_fail(True)
        assert PASS_SYMBOL in result
        assert "PASS" in result
        assert "\033[92m" in result

    def test_default_args_fail(self) -> None:
        """Test default arguments for fail (color and symbols enabled)."""
        result = format_pass_fail(False)
        assert FAIL_SYMBOL in result
        assert "FAIL" in result
        assert "\033[91m" in result


# ============================================================================
# Tests for generate_alt_text (ACC-002)
# ============================================================================


@pytest.mark.unit
@pytest.mark.visualization
class TestGenerateAltText:
    """Tests for generate_alt_text function."""

    def test_with_array_and_title(self, sample_data: np.ndarray) -> None:
        """Test alt-text generation from array data with title."""
        result = generate_alt_text(sample_data, "waveform", title="Test Signal")
        assert "Test Signal" in result
        assert "Waveform plot" in result
        assert "Time vs Amplitude" in result
        assert f"Contains {len(sample_data)} samples" in result

    def test_with_array_no_title(self, sample_data: np.ndarray) -> None:
        """Test alt-text generation from array data without title."""
        result = generate_alt_text(sample_data, "waveform")
        assert "Waveform plot" in result
        assert "Time vs Amplitude" in result
        assert "Test Signal" not in result

    def test_with_stats_dict(self, sample_stats: dict[str, float]) -> None:
        """Test alt-text generation from statistics dictionary."""
        result = generate_alt_text(sample_stats, "histogram")
        assert "Histogram plot" in result
        assert f"Contains {sample_stats['n_samples']} samples" in result
        assert "Range: -2" in result
        assert "to 3" in result
        assert "Mean: 0.5" in result
        assert "Standard deviation: 1.5" in result

    def test_with_partial_stats_dict_no_n_samples(self) -> None:
        """Test with stats dict missing n_samples."""
        stats = {"min": -2.0, "max": 3.0, "mean": 0.5, "std": 1.5}
        result = generate_alt_text(stats, "histogram")
        assert "Histogram plot" in result
        assert "Range: -2" in result
        assert "to 3" in result
        assert "Mean: 0.5" in result
        assert "Standard deviation: 1.5" in result
        # Should not have samples count
        assert "samples" not in result.lower()

    def test_with_partial_stats_dict_no_min_max(self) -> None:
        """Test with stats dict missing min/max."""
        stats = {"mean": 0.5, "std": 1.5, "n_samples": 10}
        result = generate_alt_text(stats, "histogram")
        assert "Contains 10 samples" in result
        assert "Mean: 0.5" in result
        assert "Standard deviation: 1.5" in result
        # Should not have range
        assert "Range:" not in result

    def test_with_partial_stats_dict_no_mean(self) -> None:
        """Test with stats dict missing mean."""
        stats = {"min": -2.0, "max": 3.0, "std": 1.5, "n_samples": 10}
        result = generate_alt_text(stats, "histogram")
        assert "Contains 10 samples" in result
        assert "Range: -2" in result
        assert "to 3" in result
        assert "Standard deviation: 1.5" in result
        # Should not have mean
        assert "Mean:" not in result

    def test_with_partial_stats_dict_no_std(self) -> None:
        """Test with stats dict missing std."""
        stats = {"min": -2.0, "max": 3.0, "mean": 0.5, "n_samples": 10}
        result = generate_alt_text(stats, "histogram")
        assert "Contains 10 samples" in result
        assert "Range: -2" in result
        assert "to 3" in result
        assert "Mean: 0.5" in result
        # Should not have std
        assert "Standard deviation:" not in result

    def test_with_empty_stats_dict(self) -> None:
        """Test with empty stats dict."""
        stats: dict[str, float] = {}
        result = generate_alt_text(stats, "histogram")
        assert "Histogram plot" in result
        assert "Time vs Amplitude" in result
        # Should not have any stats
        assert "Contains" not in result
        assert "Range:" not in result
        assert "Mean:" not in result
        assert "Standard deviation:" not in result

    def test_custom_axis_labels(self, sample_data: np.ndarray) -> None:
        """Test with custom x and y axis labels."""
        result = generate_alt_text(sample_data, "spectrum", x_label="Frequency", y_label="Power")
        assert "Frequency vs Power" in result

    def test_plot_type_formatting(self, sample_data: np.ndarray) -> None:
        """Test that plot type is formatted correctly."""
        result = generate_alt_text(sample_data, "eye_diagram")
        assert "Eye Diagram plot" in result  # Underscores replaced, title-cased

    def test_with_sample_rate_nanoseconds(self, sample_data: np.ndarray) -> None:
        """Test duration calculation with nanosecond duration."""
        sample_rate = 10e9  # 10 GHz
        result = generate_alt_text(sample_data, "waveform", sample_rate=sample_rate)
        assert "ns" in result
        assert "Duration:" in result

    def test_with_sample_rate_microseconds(self, sample_data: np.ndarray) -> None:
        """Test duration calculation with microsecond duration."""
        sample_rate = 1e6  # 1 MHz
        result = generate_alt_text(sample_data, "waveform", sample_rate=sample_rate)
        assert "µs" in result
        assert "Duration:" in result

    def test_with_sample_rate_milliseconds(self, sample_data: np.ndarray) -> None:
        """Test duration calculation with millisecond duration."""
        sample_rate = 1e3  # 1 kHz
        result = generate_alt_text(sample_data, "waveform", sample_rate=sample_rate)
        assert "ms" in result
        assert "Duration:" in result

    def test_with_sample_rate_seconds(self, sample_data: np.ndarray) -> None:
        """Test duration calculation with second duration."""
        sample_rate = 1.0  # 1 Hz
        result = generate_alt_text(sample_data, "waveform", sample_rate=sample_rate)
        assert " s" in result  # Space before 's' to distinguish from 'ms'
        assert "Duration:" in result

    def test_statistics_calculation(self) -> None:
        """Test that statistics are calculated correctly from array."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = generate_alt_text(data, "waveform")

        assert "Contains 5 samples" in result
        assert "Range: 1 to 5" in result
        assert "Mean: 3" in result

    def test_with_negative_values(self) -> None:
        """Test with negative values in data."""
        data = np.array([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
        result = generate_alt_text(data, "waveform")

        assert "Range: -5 to 5" in result
        assert "Mean: 0" in result

    def test_with_constant_data(self) -> None:
        """Test with constant data (std=0)."""
        data = np.array([2.5, 2.5, 2.5, 2.5])
        result = generate_alt_text(data, "waveform")

        assert "Mean: 2.5" in result
        assert "Standard deviation: 0" in result

    def test_complete_example(self) -> None:
        """Test complete example with all parameters."""
        data = np.sin(2 * np.pi * 1e3 * np.linspace(0, 1e-3, 1000))
        result = generate_alt_text(
            data,
            "waveform",
            title="1 kHz sine wave",
            x_label="Time",
            y_label="Voltage",
            sample_rate=1e6,
        )

        assert "1 kHz sine wave" in result
        assert "Waveform plot" in result
        assert "Time vs Voltage" in result
        assert "Contains 1000 samples" in result
        assert "Duration:" in result
        assert "ms" in result  # 1 ms duration (1000 samples / 1e6 Hz = 1e-3 s)


# ============================================================================
# Tests for add_plot_aria_attributes (ACC-002)
# ============================================================================


@pytest.mark.unit
@pytest.mark.visualization
class TestAddPlotAriaAttributes:
    """Tests for add_plot_aria_attributes function."""

    def test_basic_attributes(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test adding basic ARIA attributes."""
        fig, _ = fig_and_ax
        alt_text = "Test plot description"

        add_plot_aria_attributes(fig, alt_text)

        assert hasattr(fig, "_oscura_accessibility")
        assert fig._oscura_accessibility["alt_text"] == alt_text
        assert fig._oscura_accessibility["aria_role"] == "img"

    def test_custom_role(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test adding custom ARIA role."""
        fig, _ = fig_and_ax
        alt_text = "Interactive plot"

        add_plot_aria_attributes(fig, alt_text, role="application")

        assert fig._oscura_accessibility["aria_role"] == "application"

    def test_with_label(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test adding ARIA label."""
        fig, _ = fig_and_ax
        alt_text = "Plot description"
        label = "Signal waveform"

        add_plot_aria_attributes(fig, alt_text, label=label)

        assert fig._oscura_accessibility["aria_label"] == label

    def test_without_label(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test without ARIA label."""
        fig, _ = fig_and_ax

        add_plot_aria_attributes(fig, "Description")

        assert "aria_label" not in fig._oscura_accessibility

    def test_overwrites_existing_attributes(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test that adding attributes twice overwrites the first."""
        fig, _ = fig_and_ax

        add_plot_aria_attributes(fig, "First description")
        add_plot_aria_attributes(fig, "Second description", role="figure")

        assert fig._oscura_accessibility["alt_text"] == "Second description"
        assert fig._oscura_accessibility["aria_role"] == "figure"

    def test_preserves_other_metadata(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test that existing accessibility metadata is preserved."""
        fig, _ = fig_and_ax

        add_plot_aria_attributes(fig, "Description 1", label="Label 1")
        add_plot_aria_attributes(fig, "Description 2")

        # New alt_text, but label should be preserved
        assert fig._oscura_accessibility["alt_text"] == "Description 2"
        assert fig._oscura_accessibility["aria_label"] == "Label 1"


# ============================================================================
# Tests for KeyboardHandler class (ACC-003)
# ============================================================================


@pytest.mark.unit
@pytest.mark.visualization
class TestKeyboardHandlerInit:
    """Tests for KeyboardHandler initialization."""

    def test_initialization(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test basic initialization."""
        fig, ax = fig_and_ax
        handler = KeyboardHandler(fig, ax)

        assert handler.fig is fig
        assert handler.axes is ax
        assert handler.cursor_position == 0.0
        assert handler.cursor_line is None
        assert handler.enabled is False
        assert handler._connection_id is None

    def test_callback_initialization(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test that callbacks are initially None."""
        fig, ax = fig_and_ax
        handler = KeyboardHandler(fig, ax)

        assert handler.on_cursor_move is None
        assert handler.on_select is None
        assert handler.on_escape is None


@pytest.mark.unit
@pytest.mark.visualization
class TestKeyboardHandlerEnable:
    """Tests for KeyboardHandler.enable()."""

    def test_enable(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test enabling keyboard navigation."""
        fig, ax = fig_and_ax
        handler = KeyboardHandler(fig, ax)

        handler.enable()

        assert handler.enabled is True
        assert handler._connection_id is not None

    def test_enable_idempotent(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test that calling enable multiple times is safe."""
        fig, ax = fig_and_ax
        handler = KeyboardHandler(fig, ax)

        handler.enable()
        first_id = handler._connection_id

        handler.enable()
        second_id = handler._connection_id

        # Should not reconnect if already enabled
        assert first_id == second_id


@pytest.mark.unit
@pytest.mark.visualization
class TestKeyboardHandlerDisable:
    """Tests for KeyboardHandler.disable()."""

    def test_disable(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test disabling keyboard navigation."""
        fig, ax = fig_and_ax
        handler = KeyboardHandler(fig, ax)

        handler.enable()
        handler.disable()

        assert handler.enabled is False
        assert handler._connection_id is None

    def test_disable_when_not_enabled(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test disabling when not enabled."""
        fig, ax = fig_and_ax
        handler = KeyboardHandler(fig, ax)

        # Should not raise error
        handler.disable()

        assert handler.enabled is False


@pytest.mark.unit
@pytest.mark.visualization
class TestKeyboardHandlerKeyEvents:
    """Tests for KeyboardHandler keyboard event handling."""

    def test_arrow_left(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test left arrow key moves cursor left."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        handler = KeyboardHandler(fig, ax)
        handler.cursor_position = 50.0

        # Create mock event
        event = Mock()
        event.key = "left"

        handler._on_key_press(event)

        assert handler.cursor_position < 50.0

    def test_arrow_right(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test right arrow key moves cursor right."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        handler = KeyboardHandler(fig, ax)
        handler.cursor_position = 50.0

        event = Mock()
        event.key = "right"

        handler._on_key_press(event)

        assert handler.cursor_position > 50.0

    def test_enter_triggers_select(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test enter key triggers select callback."""
        fig, ax = fig_and_ax
        handler = KeyboardHandler(fig, ax)

        callback_called = False

        def on_select() -> None:
            nonlocal callback_called
            callback_called = True

        handler.on_select = on_select

        event = Mock()
        event.key = "enter"
        handler._on_key_press(event)

        assert callback_called

    def test_escape_triggers_escape(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test escape key triggers escape callback."""
        fig, ax = fig_and_ax
        handler = KeyboardHandler(fig, ax)

        callback_called = False

        def on_escape() -> None:
            nonlocal callback_called
            callback_called = True

        handler.on_escape = on_escape

        event = Mock()
        event.key = "escape"
        handler._on_key_press(event)

        assert callback_called

    def test_plus_zooms_in(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test plus key zooms in."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        handler = KeyboardHandler(fig, ax)

        initial_x_range = ax.get_xlim()[1] - ax.get_xlim()[0]

        event = Mock()
        event.key = "+"
        handler._on_key_press(event)

        new_x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        assert new_x_range < initial_x_range

    def test_equals_zooms_in(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test equals key also zooms in."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        handler = KeyboardHandler(fig, ax)

        initial_x_range = ax.get_xlim()[1] - ax.get_xlim()[0]

        event = Mock()
        event.key = "="
        handler._on_key_press(event)

        new_x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        assert new_x_range < initial_x_range

    def test_minus_zooms_out(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test minus key zooms out."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        handler = KeyboardHandler(fig, ax)

        initial_x_range = ax.get_xlim()[1] - ax.get_xlim()[0]

        event = Mock()
        event.key = "-"
        handler._on_key_press(event)

        new_x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        assert new_x_range > initial_x_range

    def test_underscore_zooms_out(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test underscore key also zooms out."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        handler = KeyboardHandler(fig, ax)

        initial_x_range = ax.get_xlim()[1] - ax.get_xlim()[0]

        event = Mock()
        event.key = "_"
        handler._on_key_press(event)

        new_x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        assert new_x_range > initial_x_range

    def test_home_jumps_to_start(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test home key jumps to start."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        handler = KeyboardHandler(fig, ax)
        handler.cursor_position = 50.0

        event = Mock()
        event.key = "home"
        handler._on_key_press(event)

        assert handler.cursor_position == 0.0

    def test_end_jumps_to_end(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test end key jumps to end."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        handler = KeyboardHandler(fig, ax)
        handler.cursor_position = 50.0

        event = Mock()
        event.key = "end"
        handler._on_key_press(event)

        assert handler.cursor_position == 100.0

    def test_home_with_callback(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test home key triggers cursor move callback."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        handler = KeyboardHandler(fig, ax)
        handler.cursor_position = 50.0

        positions = []
        handler.on_cursor_move = lambda pos: positions.append(pos)

        event = Mock()
        event.key = "home"
        handler._on_key_press(event)

        assert len(positions) == 1
        assert positions[0] == 0.0

    def test_end_with_callback(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test end key triggers cursor move callback."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        handler = KeyboardHandler(fig, ax)
        handler.cursor_position = 50.0

        positions = []
        handler.on_cursor_move = lambda pos: positions.append(pos)

        event = Mock()
        event.key = "end"
        handler._on_key_press(event)

        assert len(positions) == 1
        assert positions[0] == 100.0

    def test_none_key_ignored(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test that None key is ignored."""
        fig, ax = fig_and_ax
        handler = KeyboardHandler(fig, ax)
        initial_pos = handler.cursor_position

        event = Mock()
        event.key = None
        handler._on_key_press(event)

        # Should not change anything
        assert handler.cursor_position == initial_pos

    def test_unknown_key_ignored(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test that unknown keys are ignored."""
        fig, ax = fig_and_ax
        handler = KeyboardHandler(fig, ax)
        initial_pos = handler.cursor_position

        event = Mock()
        event.key = "x"
        handler._on_key_press(event)

        # Should not change anything
        assert handler.cursor_position == initial_pos

    def test_enter_without_callback(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test enter key when on_select callback is None."""
        fig, ax = fig_and_ax
        handler = KeyboardHandler(fig, ax)
        # Ensure on_select is None
        handler.on_select = None

        event = Mock()
        event.key = "enter"
        # Should not raise an error
        handler._on_key_press(event)

    def test_escape_without_callback(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test escape key when on_escape callback is None."""
        fig, ax = fig_and_ax
        handler = KeyboardHandler(fig, ax)
        # Ensure on_escape is None
        handler.on_escape = None

        event = Mock()
        event.key = "escape"
        # Should not raise an error
        handler._on_key_press(event)


@pytest.mark.unit
@pytest.mark.visualization
class TestKeyboardHandlerCursorMovement:
    """Tests for cursor movement details."""

    def test_cursor_move_callback(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test cursor move callback is triggered."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        handler = KeyboardHandler(fig, ax)

        positions = []

        def on_move(pos: float) -> None:
            positions.append(pos)

        handler.on_cursor_move = on_move

        event = Mock()
        event.key = "right"
        handler._on_key_press(event)

        assert len(positions) == 1
        assert positions[0] > 0

    def test_cursor_stays_in_bounds_left(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test cursor doesn't go past left boundary."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        handler = KeyboardHandler(fig, ax)
        handler.cursor_position = 0.0

        event = Mock()
        event.key = "left"
        handler._on_key_press(event)

        assert handler.cursor_position >= 0.0

    def test_cursor_stays_in_bounds_right(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test cursor doesn't go past right boundary."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        handler = KeyboardHandler(fig, ax)
        handler.cursor_position = 100.0

        event = Mock()
        event.key = "right"
        handler._on_key_press(event)

        assert handler.cursor_position <= 100.0

    def test_cursor_step_size(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test that cursor moves by 1% of range."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        handler = KeyboardHandler(fig, ax)
        handler.cursor_position = 50.0

        event = Mock()
        event.key = "right"
        handler._on_key_press(event)

        # Should move by ~1% of range (1.0)
        assert abs(handler.cursor_position - 51.0) < 0.1


@pytest.mark.unit
@pytest.mark.visualization
class TestKeyboardHandlerZoom:
    """Tests for zoom functionality."""

    def test_zoom_centers_on_middle(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test that zoom centers on the middle of the plot."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        handler = KeyboardHandler(fig, ax)

        event = Mock()
        event.key = "+"
        handler._on_key_press(event)

        xlim = ax.get_xlim()
        x_center = (xlim[0] + xlim[1]) / 2
        assert abs(x_center - 50.0) < 0.1

    def test_zoom_factor_in(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test zoom in factor is 1.2."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 120)
        handler = KeyboardHandler(fig, ax)

        event = Mock()
        event.key = "+"
        handler._on_key_press(event)

        xlim = ax.get_xlim()
        new_range = xlim[1] - xlim[0]
        expected_range = 120 / 1.2

        assert abs(new_range - expected_range) < 0.1

    def test_zoom_factor_out(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test zoom out factor is 0.8."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        handler = KeyboardHandler(fig, ax)

        event = Mock()
        event.key = "-"
        handler._on_key_press(event)

        xlim = ax.get_xlim()
        new_range = xlim[1] - xlim[0]
        expected_range = 100 / 0.8

        assert abs(new_range - expected_range) < 0.1


@pytest.mark.unit
@pytest.mark.visualization
class TestKeyboardHandlerCursorLine:
    """Tests for cursor line visualization."""

    def test_cursor_line_created_on_first_move(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test that cursor line is created on first movement."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        handler = KeyboardHandler(fig, ax)

        # Mock the canvas to prevent drawing
        with patch.object(fig.canvas, "draw_idle"):
            assert handler.cursor_line is None

            event = Mock()
            event.key = "right"
            handler._on_key_press(event)

            assert handler.cursor_line is not None

    def test_cursor_line_updated_on_subsequent_moves(
        self, fig_and_ax: tuple[Figure, plt.Axes]
    ) -> None:
        """Test that cursor line is updated, not recreated."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        handler = KeyboardHandler(fig, ax)

        # Mock the canvas to prevent drawing
        with patch.object(fig.canvas, "draw_idle"):
            event = Mock()
            event.key = "right"
            handler._on_key_press(event)

            first_line = handler.cursor_line

            event.key = "right"
            handler._on_key_press(event)

            # Same line object, just updated
            assert handler.cursor_line is first_line


@pytest.mark.unit
@pytest.mark.visualization
class TestKeyboardHandlerIntegration:
    """Integration tests for KeyboardHandler."""

    def test_complete_workflow(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test complete workflow: enable, navigate, disable."""
        fig, ax = fig_and_ax
        ax.plot([0, 1, 2, 3], [0, 1, 0, 1])
        handler = KeyboardHandler(fig, ax)

        # Enable
        handler.enable()
        assert handler.enabled

        # Navigate
        event = Mock()
        event.key = "right"
        handler._on_key_press(event)
        assert handler.cursor_position > 0

        # Disable
        handler.disable()
        assert not handler.enabled

    def test_multiple_callbacks(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test multiple callbacks can be set and triggered."""
        fig, ax = fig_and_ax
        ax.set_xlim(0, 100)
        handler = KeyboardHandler(fig, ax)

        cursor_moves = []
        selects = []
        escapes = []

        handler.on_cursor_move = lambda pos: cursor_moves.append(pos)
        handler.on_select = lambda: selects.append(True)
        handler.on_escape = lambda: escapes.append(True)

        # Trigger each callback
        event = Mock()
        event.key = "right"
        handler._on_key_press(event)

        event.key = "enter"
        handler._on_key_press(event)

        event.key = "escape"
        handler._on_key_press(event)

        assert len(cursor_moves) == 1
        assert len(selects) == 1
        assert len(escapes) == 1


# ============================================================================
# Tests for module constants
# ============================================================================


@pytest.mark.unit
@pytest.mark.visualization
class TestConstants:
    """Tests for module constants."""

    def test_line_styles(self) -> None:
        """Test LINE_STYLES constant."""
        assert LINE_STYLES == ["solid", "dashed", "dotted", "dashdot"]

    def test_pass_symbol(self) -> None:
        """Test PASS_SYMBOL constant."""
        assert PASS_SYMBOL == "✓"

    def test_fail_symbol(self) -> None:
        """Test FAIL_SYMBOL constant."""
        assert FAIL_SYMBOL == "✗"


# ============================================================================
# Integration tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.visualization
class TestAccessibilityIntegration:
    """Integration tests combining multiple accessibility features."""

    def test_complete_accessible_plot(
        self, sample_data: np.ndarray, fig_and_ax: tuple[Figure, plt.Axes]
    ) -> None:
        """Test creating a fully accessible plot."""
        fig, ax = fig_and_ax

        # Use colorblind-safe palette
        cmap_name = get_colorblind_palette("viridis")
        cmap = plt.get_cmap(cmap_name)

        # Plot with accessible colors
        ax.plot(sample_data, color=cmap(0.5))

        # Generate alt-text
        alt_text = generate_alt_text(sample_data, "waveform", title="Test Signal", sample_rate=1e6)

        # Add ARIA attributes
        add_plot_aria_attributes(fig, alt_text, label="Test Waveform")

        # Add keyboard navigation
        handler = KeyboardHandler(fig, ax)
        handler.enable()

        # Verify all components
        assert fig._oscura_accessibility["alt_text"] == alt_text
        assert handler.enabled
        assert cmap_name == "viridis"

        handler.disable()

    def test_multi_line_plot_with_accessibility(self, fig_and_ax: tuple[Figure, plt.Axes]) -> None:
        """Test multi-line plot with accessible styling."""
        fig, ax = fig_and_ax

        # Generate multiple signals
        n_lines = 4
        signals = [np.sin(2 * np.pi * (i + 1) * np.linspace(0, 1, 100)) for i in range(n_lines)]

        # Get accessible styles
        styles = get_multi_line_styles(n_lines)

        # Plot with distinct styles
        for signal, (color, linestyle) in zip(signals, styles, strict=False):
            ax.plot(signal, color=color, linestyle=linestyle)

        # Generate combined alt-text
        all_data = np.concatenate(signals)
        alt_text = generate_alt_text(all_data, "waveform", title="Multi-channel Signal")

        # Add ARIA attributes
        add_plot_aria_attributes(fig, alt_text)

        # Verify
        assert len(styles) == n_lines
        assert fig._oscura_accessibility["alt_text"] == alt_text
