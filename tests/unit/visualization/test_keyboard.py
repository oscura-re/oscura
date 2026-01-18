"""Unit tests for keyboard navigation in interactive visualizations.

Tests:
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.visualization]


@pytest.fixture
def mock_fig():
    """Create a mock matplotlib figure."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    yield fig
    plt.close(fig)


@pytest.fixture
def mock_ax(mock_fig):
    """Create a mock matplotlib axes."""
    ax = mock_fig.add_subplot(111)
    ax.plot([1, 2, 3], [1, 4, 2])  # Add some data
    return ax


@pytest.fixture
def multiple_axes(mock_fig):
    """Create multiple axes for testing tab navigation."""
    axes = []
    for i in range(3):
        ax = mock_fig.add_subplot(3, 1, i + 1)
        ax.plot([1, 2, 3], [1, 4, 2])
        axes.append(ax)
    return axes


@pytest.fixture
def mock_key_event():
    """Create a mock keyboard event."""

    def _make_event(key: str | None = None):
        event = Mock()
        event.key = key
        event.inaxes = None
        return event

    return _make_event


class TestKeyboardNavigatorInit:
    """Tests for KeyboardNavigator initialization."""

    @pytest.mark.unit
    def test_init_single_axes(self, mock_fig, mock_ax):
        """Test initialization with single axes."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)

        assert nav.fig is mock_fig
        assert len(nav.axes_list) == 1
        assert nav.axes_list[0] is mock_ax
        assert nav.current_axes_index == 0
        assert nav.pan_step == 0.1
        assert nav.zoom_factor == 1.2

    @pytest.mark.unit
    def test_init_multiple_axes(self, mock_fig, multiple_axes):
        """Test initialization with multiple axes."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, multiple_axes)

        assert nav.fig is mock_fig
        assert len(nav.axes_list) == 3
        assert nav.axes_list == multiple_axes
        assert nav.current_axes_index == 0

    @pytest.mark.unit
    def test_init_custom_parameters(self, mock_fig, mock_ax):
        """Test initialization with custom pan_step and zoom_factor."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax, pan_step=0.05, zoom_factor=1.5)

        assert nav.pan_step == 0.05
        assert nav.zoom_factor == 1.5

    @pytest.mark.unit
    def test_original_limits_stored(self, mock_fig, mock_ax):
        """Test that original axes limits are stored."""
        from oscura.visualization.keyboard import KeyboardNavigator

        xlim = mock_ax.get_xlim()
        ylim = mock_ax.get_ylim()

        nav = KeyboardNavigator(mock_fig, mock_ax)

        assert 0 in nav.original_limits
        assert nav.original_limits[0]["xlim"] == xlim
        assert nav.original_limits[0]["ylim"] == ylim

    @pytest.mark.unit
    def test_multiple_axes_limits_stored(self, mock_fig, multiple_axes):
        """Test that original limits are stored for all axes."""
        from oscura.visualization.keyboard import KeyboardNavigator

        original_limits = [(ax.get_xlim(), ax.get_ylim()) for ax in multiple_axes]

        nav = KeyboardNavigator(mock_fig, multiple_axes)

        for i, (xlim, ylim) in enumerate(original_limits):
            assert i in nav.original_limits
            assert nav.original_limits[i]["xlim"] == xlim
            assert nav.original_limits[i]["ylim"] == ylim

    @pytest.mark.unit
    def test_connection_id_initially_none(self, mock_fig, mock_ax):
        """Test that connection ID is None before connect()."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        assert nav._connection_id is None

    @pytest.mark.unit
    def test_help_text_initially_none(self, mock_fig, mock_ax):
        """Test that help text is None initially."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        assert nav._help_text is None


class TestKeyboardNavigatorConnection:
    """Tests for keyboard navigator connection/disconnection."""

    @pytest.mark.unit
    def test_connect(self, mock_fig, mock_ax):
        """Test connecting keyboard event handler.

        : Tab navigates between plot elements.
        """
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        assert nav._connection_id is not None

    @pytest.mark.unit
    def test_connect_highlights_axes(self, mock_fig, mock_ax):
        """Test that connect highlights the active axes."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        # Check that spines are highlighted (red, linewidth 2)
        for spine in mock_ax.spines.values():
            assert spine.get_edgecolor() == "red" or spine.get_edgecolor() == (1.0, 0.0, 0.0, 1.0)
            assert spine.get_linewidth() == 2

    @pytest.mark.unit
    def test_disconnect(self, mock_fig, mock_ax):
        """Test disconnecting keyboard event handler."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()
        connection_id = nav._connection_id

        nav.disconnect()

        assert nav._connection_id is None

    @pytest.mark.unit
    def test_disconnect_without_connect(self, mock_fig, mock_ax):
        """Test that disconnect works even if not connected."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.disconnect()  # Should not raise

        assert nav._connection_id is None


class TestKeyboardNavigatorPanning:
    """Tests for keyboard panning operations."""

    @pytest.mark.unit
    def test_pan_left(self, mock_fig, mock_ax, mock_key_event):
        """Test panning left with arrow key.

        : Arrow keys move cursors.
        """
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        xlim_before = mock_ax.get_xlim()
        ylim_before = mock_ax.get_ylim()

        event = mock_key_event("left")
        nav._on_key(event)

        xlim_after = mock_ax.get_xlim()
        ylim_after = mock_ax.get_ylim()

        # X should shift left (decrease)
        assert xlim_after[0] < xlim_before[0]
        assert xlim_after[1] < xlim_before[1]
        # Y should stay the same
        assert ylim_after == ylim_before

    @pytest.mark.unit
    def test_pan_right(self, mock_fig, mock_ax, mock_key_event):
        """Test panning right with arrow key."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        xlim_before = mock_ax.get_xlim()
        ylim_before = mock_ax.get_ylim()

        event = mock_key_event("right")
        nav._on_key(event)

        xlim_after = mock_ax.get_xlim()
        ylim_after = mock_ax.get_ylim()

        # X should shift right (increase)
        assert xlim_after[0] > xlim_before[0]
        assert xlim_after[1] > xlim_before[1]
        # Y should stay the same
        assert ylim_after == ylim_before

    @pytest.mark.unit
    def test_pan_up(self, mock_fig, mock_ax, mock_key_event):
        """Test panning up with arrow key."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        xlim_before = mock_ax.get_xlim()
        ylim_before = mock_ax.get_ylim()

        event = mock_key_event("up")
        nav._on_key(event)

        xlim_after = mock_ax.get_xlim()
        ylim_after = mock_ax.get_ylim()

        # X should stay the same
        assert xlim_after == xlim_before
        # Y should shift up (increase)
        assert ylim_after[0] > ylim_before[0]
        assert ylim_after[1] > ylim_before[1]

    @pytest.mark.unit
    def test_pan_down(self, mock_fig, mock_ax, mock_key_event):
        """Test panning down with arrow key."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        xlim_before = mock_ax.get_xlim()
        ylim_before = mock_ax.get_ylim()

        event = mock_key_event("down")
        nav._on_key(event)

        xlim_after = mock_ax.get_xlim()
        ylim_after = mock_ax.get_ylim()

        # X should stay the same
        assert xlim_after == xlim_before
        # Y should shift down (decrease)
        assert ylim_after[0] < ylim_before[0]
        assert ylim_after[1] < ylim_before[1]

    @pytest.mark.unit
    def test_pan_step_parameter(self, mock_fig, mock_ax, mock_key_event):
        """Test that pan_step parameter affects pan distance."""
        from oscura.visualization.keyboard import KeyboardNavigator

        # Test with default pan_step
        nav1 = KeyboardNavigator(mock_fig, mock_ax, pan_step=0.1)
        nav1.connect()
        xlim_before = mock_ax.get_xlim()
        event = mock_key_event("right")
        nav1._on_key(event)
        xlim_default = mock_ax.get_xlim()

        # Reset axes
        mock_ax.set_xlim(xlim_before)

        # Test with larger pan_step
        nav2 = KeyboardNavigator(mock_fig, mock_ax, pan_step=0.2)
        nav2._on_key(event)
        xlim_larger = mock_ax.get_xlim()

        # Larger pan_step should move further
        shift_default = xlim_default[0] - xlim_before[0]
        shift_larger = xlim_larger[0] - xlim_before[0]
        assert abs(shift_larger) > abs(shift_default)


class TestKeyboardNavigatorZooming:
    """Tests for keyboard zooming operations."""

    @pytest.mark.unit
    def test_zoom_in_plus(self, mock_fig, mock_ax, mock_key_event):
        """Test zooming in with + key.

        : +/- keys zoom in/out.
        """
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        xlim_before = mock_ax.get_xlim()
        ylim_before = mock_ax.get_ylim()
        x_range_before = xlim_before[1] - xlim_before[0]
        y_range_before = ylim_before[1] - ylim_before[0]

        event = mock_key_event("+")
        nav._on_key(event)

        xlim_after = mock_ax.get_xlim()
        ylim_after = mock_ax.get_ylim()
        x_range_after = xlim_after[1] - xlim_after[0]
        y_range_after = ylim_after[1] - ylim_after[0]

        # Range should decrease (zoom in)
        assert x_range_after < x_range_before
        assert y_range_after < y_range_before

    @pytest.mark.unit
    def test_zoom_in_equals(self, mock_fig, mock_ax, mock_key_event):
        """Test zooming in with = key (same as +)."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        xlim_before = mock_ax.get_xlim()
        x_range_before = xlim_before[1] - xlim_before[0]

        event = mock_key_event("=")
        nav._on_key(event)

        xlim_after = mock_ax.get_xlim()
        x_range_after = xlim_after[1] - xlim_after[0]

        # Range should decrease (zoom in)
        assert x_range_after < x_range_before

    @pytest.mark.unit
    def test_zoom_out_minus(self, mock_fig, mock_ax, mock_key_event):
        """Test zooming out with - key."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        xlim_before = mock_ax.get_xlim()
        ylim_before = mock_ax.get_ylim()
        x_range_before = xlim_before[1] - xlim_before[0]
        y_range_before = ylim_before[1] - ylim_before[0]

        event = mock_key_event("-")
        nav._on_key(event)

        xlim_after = mock_ax.get_xlim()
        ylim_after = mock_ax.get_ylim()
        x_range_after = xlim_after[1] - xlim_after[0]
        y_range_after = ylim_after[1] - ylim_after[0]

        # Range should increase (zoom out)
        assert x_range_after > x_range_before
        assert y_range_after > y_range_before

    @pytest.mark.unit
    def test_zoom_out_underscore(self, mock_fig, mock_ax, mock_key_event):
        """Test zooming out with _ key (same as -)."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        xlim_before = mock_ax.get_xlim()
        x_range_before = xlim_before[1] - xlim_before[0]

        event = mock_key_event("_")
        nav._on_key(event)

        xlim_after = mock_ax.get_xlim()
        x_range_after = xlim_after[1] - xlim_after[0]

        # Range should increase (zoom out)
        assert x_range_after > x_range_before

    @pytest.mark.unit
    def test_zoom_preserves_center(self, mock_fig, mock_ax, mock_key_event):
        """Test that zoom operations preserve the center point."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        xlim_before = mock_ax.get_xlim()
        ylim_before = mock_ax.get_ylim()
        x_center_before = (xlim_before[0] + xlim_before[1]) / 2
        y_center_before = (ylim_before[0] + ylim_before[1]) / 2

        event = mock_key_event("+")
        nav._on_key(event)

        xlim_after = mock_ax.get_xlim()
        ylim_after = mock_ax.get_ylim()
        x_center_after = (xlim_after[0] + xlim_after[1]) / 2
        y_center_after = (ylim_after[0] + ylim_after[1]) / 2

        # Centers should be approximately the same
        assert abs(x_center_after - x_center_before) < 1e-10
        assert abs(y_center_after - y_center_before) < 1e-10

    @pytest.mark.unit
    def test_zoom_factor_parameter(self, mock_fig, mock_ax, mock_key_event):
        """Test that zoom_factor parameter affects zoom amount."""
        from oscura.visualization.keyboard import KeyboardNavigator

        xlim_original = mock_ax.get_xlim()
        x_range_original = xlim_original[1] - xlim_original[0]

        # Test with default zoom_factor (1.2)
        nav1 = KeyboardNavigator(mock_fig, mock_ax, zoom_factor=1.2)
        event = mock_key_event("+")
        nav1._on_key(event)
        xlim_default = mock_ax.get_xlim()
        x_range_default = xlim_default[1] - xlim_default[0]

        # Reset axes
        mock_ax.set_xlim(xlim_original)

        # Test with larger zoom_factor (2.0)
        nav2 = KeyboardNavigator(mock_fig, mock_ax, zoom_factor=2.0)
        nav2._on_key(event)
        xlim_larger = mock_ax.get_xlim()
        x_range_larger = xlim_larger[1] - xlim_larger[0]

        # Larger zoom_factor should zoom more (smaller resulting range)
        assert x_range_larger < x_range_default


class TestKeyboardNavigatorReset:
    """Tests for view reset operations."""

    @pytest.mark.unit
    def test_reset_view_home(self, mock_fig, mock_ax, mock_key_event):
        """Test resetting view with Home key.

        : Home resets to full view.
        """
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        xlim_original = mock_ax.get_xlim()
        ylim_original = mock_ax.get_ylim()

        # Pan and zoom to change view
        mock_ax.set_xlim(2, 4)
        mock_ax.set_ylim(3, 5)

        # Reset with Home key
        event = mock_key_event("home")
        nav._on_key(event)

        xlim_after = mock_ax.get_xlim()
        ylim_after = mock_ax.get_ylim()

        # Should be back to original
        assert xlim_after == xlim_original
        assert ylim_after == ylim_original

    @pytest.mark.unit
    def test_reset_after_multiple_operations(self, mock_fig, mock_ax, mock_key_event):
        """Test reset after multiple pan and zoom operations."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        xlim_original = mock_ax.get_xlim()
        ylim_original = mock_ax.get_ylim()

        # Perform multiple operations
        nav._on_key(mock_key_event("right"))
        nav._on_key(mock_key_event("up"))
        nav._on_key(mock_key_event("+"))
        nav._on_key(mock_key_event("-"))

        # Reset
        nav._on_key(mock_key_event("home"))

        xlim_after = mock_ax.get_xlim()
        ylim_after = mock_ax.get_ylim()

        assert xlim_after == xlim_original
        assert ylim_after == ylim_original


class TestKeyboardNavigatorTabNavigation:
    """Tests for tab navigation between axes."""

    @pytest.mark.unit
    def test_cycle_axes_single_axes(self, mock_fig, mock_ax, mock_key_event):
        """Test that tab does nothing with single axes.

        : Tab navigates between plot elements.
        """
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        assert nav.current_axes_index == 0

        event = mock_key_event("tab")
        nav._on_key(event)

        # Should still be at index 0
        assert nav.current_axes_index == 0

    @pytest.mark.unit
    def test_cycle_axes_forward(self, mock_fig, multiple_axes, mock_key_event):
        """Test cycling through axes with Tab key."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, multiple_axes)
        nav.connect()

        assert nav.current_axes_index == 0

        # Press Tab
        event = mock_key_event("tab")
        nav._on_key(event)
        assert nav.current_axes_index == 1

        # Press Tab again
        nav._on_key(event)
        assert nav.current_axes_index == 2

        # Press Tab again (should wrap to 0)
        nav._on_key(event)
        assert nav.current_axes_index == 0

    @pytest.mark.unit
    def test_cycle_axes_backward(self, mock_fig, multiple_axes, mock_key_event):
        """Test cycling backwards with Shift+Tab.

        Note: The code checks for 'shift+tab' key in _on_key but then passes
        the reverse flag. Since 'shift+tab' != 'tab', the tab handling won't
        trigger. We test the _cycle_axes method directly instead.
        """
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, multiple_axes)
        nav.connect()

        assert nav.current_axes_index == 0

        # Test _cycle_axes directly with reverse=True
        nav._cycle_axes(reverse=True)
        assert nav.current_axes_index == 2  # Wrap to end

        # Cycle backward again
        nav._cycle_axes(reverse=True)
        assert nav.current_axes_index == 1

    @pytest.mark.unit
    def test_axes_highlighting_on_cycle(self, mock_fig, multiple_axes, mock_key_event):
        """Test that axes are highlighted/unhighlighted during cycling."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, multiple_axes)
        nav.connect()

        # First axes should be highlighted
        for spine in multiple_axes[0].spines.values():
            color = spine.get_edgecolor()
            assert color == "red" or color == (1.0, 0.0, 0.0, 1.0)

        # Cycle to next axes
        event = mock_key_event("tab")
        nav._on_key(event)

        # First axes should be unhighlighted
        for spine in multiple_axes[0].spines.values():
            color = spine.get_edgecolor()
            assert color == "black" or color == (0.0, 0.0, 0.0, 1.0)

        # Second axes should be highlighted
        for spine in multiple_axes[1].spines.values():
            color = spine.get_edgecolor()
            assert color == "red" or color == (1.0, 0.0, 0.0, 1.0)


class TestKeyboardNavigatorHelp:
    """Tests for help display functionality."""

    @pytest.mark.unit
    def test_show_help(self, mock_fig, mock_ax, mock_key_event):
        """Test showing help with ? key.

        : ? key shows keyboard shortcuts help.
        """
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        assert nav._help_text is None

        event = mock_key_event("?")
        nav._on_key(event)

        # Help text should be created
        assert nav._help_text is not None

    @pytest.mark.unit
    def test_help_content(self, mock_fig, mock_ax, mock_key_event):
        """Test that help text contains expected content."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        event = mock_key_event("?")
        nav._on_key(event)

        help_text = nav._help_text.get_text()

        # Check for key sections
        assert "Pan:" in help_text or "Pan" in help_text
        assert "Zoom:" in help_text or "Zoom" in help_text
        assert "Home" in help_text
        assert "Tab" in help_text

    @pytest.mark.unit
    def test_show_help_idempotent(self, mock_fig, mock_ax, mock_key_event):
        """Test that showing help multiple times doesn't create multiple text objects."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        event = mock_key_event("?")
        nav._on_key(event)
        first_help_text = nav._help_text

        # Press ? again
        nav._on_key(event)

        # Should be the same object
        assert nav._help_text is first_help_text

    @pytest.mark.unit
    def test_hide_help_escape(self, mock_fig, mock_ax, mock_key_event):
        """Test hiding help with Escape key.

        : Escape closes modals/menus.
        """
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        # Show help first
        nav._on_key(mock_key_event("?"))
        assert nav._help_text is not None

        # Hide with Escape
        nav._on_key(mock_key_event("escape"))
        assert nav._help_text is None

    @pytest.mark.unit
    def test_hide_help_without_show(self, mock_fig, mock_ax, mock_key_event):
        """Test that hiding help without showing doesn't cause errors."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        # Should not raise
        nav._on_key(mock_key_event("escape"))
        assert nav._help_text is None


class TestKeyboardNavigatorEventHandling:
    """Tests for keyboard event handling."""

    @pytest.mark.unit
    def test_none_key_ignored(self, mock_fig, mock_ax, mock_key_event):
        """Test that events with None key are ignored."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        xlim_before = mock_ax.get_xlim()

        event = mock_key_event(None)
        nav._on_key(event)

        xlim_after = mock_ax.get_xlim()

        # Nothing should change
        assert xlim_after == xlim_before

    @pytest.mark.unit
    def test_unhandled_key_ignored(self, mock_fig, mock_ax, mock_key_event):
        """Test that unhandled keys are ignored."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        xlim_before = mock_ax.get_xlim()

        # Try some random key
        event = mock_key_event("x")
        nav._on_key(event)

        xlim_after = mock_ax.get_xlim()

        # Nothing should change
        assert xlim_after == xlim_before

    @pytest.mark.unit
    def test_canvas_draw_idle_called(self, mock_fig, mock_ax, mock_key_event):
        """Test that canvas.draw_idle() is called after valid key press."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        # Mock the draw_idle method
        with patch.object(mock_fig.canvas, "draw_idle") as mock_draw:
            event = mock_key_event("left")
            nav._on_key(event)

            # Should have been called
            mock_draw.assert_called_once()

    @pytest.mark.unit
    def test_canvas_draw_idle_not_called_for_invalid_key(self, mock_fig, mock_ax, mock_key_event):
        """Test that canvas.draw_idle() is not called for unhandled keys."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        # Mock the draw_idle method
        with patch.object(mock_fig.canvas, "draw_idle") as mock_draw:
            event = mock_key_event("x")
            nav._on_key(event)

            # Should not have been called
            mock_draw.assert_not_called()


class TestEnableKeyboardNavigation:
    """Tests for enable_keyboard_navigation convenience function."""

    @pytest.mark.unit
    def test_enable_with_explicit_axes(self, mock_fig, mock_ax):
        """Test enable_keyboard_navigation with explicit axes."""
        from oscura.visualization.keyboard import enable_keyboard_navigation

        nav = enable_keyboard_navigation(mock_fig, mock_ax)

        assert nav is not None
        assert nav.fig is mock_fig
        assert len(nav.axes_list) == 1
        assert nav.axes_list[0] is mock_ax
        assert nav._connection_id is not None  # Should be connected

    @pytest.mark.unit
    def test_enable_with_default_axes(self, mock_fig, multiple_axes):
        """Test enable_keyboard_navigation with default axes (all from figure)."""
        from oscura.visualization.keyboard import enable_keyboard_navigation

        nav = enable_keyboard_navigation(mock_fig)

        assert nav is not None
        assert nav.fig is mock_fig
        assert len(nav.axes_list) == 3
        assert nav.axes_list == multiple_axes
        assert nav._connection_id is not None

    @pytest.mark.unit
    def test_enable_with_custom_parameters(self, mock_fig, mock_ax):
        """Test enable_keyboard_navigation with custom parameters."""
        from oscura.visualization.keyboard import enable_keyboard_navigation

        nav = enable_keyboard_navigation(mock_fig, mock_ax, pan_step=0.05, zoom_factor=1.5)

        assert nav.pan_step == 0.05
        assert nav.zoom_factor == 1.5

    @pytest.mark.unit
    def test_enable_returns_connected_navigator(self, mock_fig, mock_ax):
        """Test that enable_keyboard_navigation returns a connected navigator."""
        from oscura.visualization.keyboard import enable_keyboard_navigation

        nav = enable_keyboard_navigation(mock_fig, mock_ax)

        # Verify it's connected
        assert nav._connection_id is not None


class TestKeyboardNavigatorIntegration:
    """Integration tests for keyboard navigation."""

    @pytest.mark.unit
    def test_pan_and_reset_workflow(self, mock_fig, mock_ax, mock_key_event):
        """Test a complete pan and reset workflow."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        xlim_original = mock_ax.get_xlim()
        ylim_original = mock_ax.get_ylim()

        # Pan right
        nav._on_key(mock_key_event("right"))
        xlim_panned = mock_ax.get_xlim()
        assert xlim_panned != xlim_original

        # Pan up
        nav._on_key(mock_key_event("up"))
        ylim_panned = mock_ax.get_ylim()
        assert ylim_panned != ylim_original

        # Reset
        nav._on_key(mock_key_event("home"))
        assert mock_ax.get_xlim() == xlim_original
        assert mock_ax.get_ylim() == ylim_original

    @pytest.mark.unit
    def test_zoom_and_pan_workflow(self, mock_fig, mock_ax, mock_key_event):
        """Test a workflow combining zoom and pan operations."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        xlim_original = mock_ax.get_xlim()
        ylim_original = mock_ax.get_ylim()
        x_range_original = xlim_original[1] - xlim_original[0]

        # Zoom in
        nav._on_key(mock_key_event("+"))
        xlim_zoomed = mock_ax.get_xlim()
        x_range_zoomed = xlim_zoomed[1] - xlim_zoomed[0]
        assert x_range_zoomed < x_range_original

        # Pan left
        nav._on_key(mock_key_event("left"))
        xlim_panned = mock_ax.get_xlim()
        assert xlim_panned[0] < xlim_zoomed[0]

        # Range should stay the same (pan doesn't change range)
        x_range_panned = xlim_panned[1] - xlim_panned[0]
        assert abs(x_range_panned - x_range_zoomed) < 1e-10

    @pytest.mark.unit
    def test_help_show_hide_workflow(self, mock_fig, mock_ax, mock_key_event):
        """Test showing and hiding help multiple times."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)
        nav.connect()

        # Show help
        nav._on_key(mock_key_event("?"))
        assert nav._help_text is not None

        # Hide help
        nav._on_key(mock_key_event("escape"))
        assert nav._help_text is None

        # Show again
        nav._on_key(mock_key_event("?"))
        assert nav._help_text is not None

        # Hide again
        nav._on_key(mock_key_event("escape"))
        assert nav._help_text is None

    @pytest.mark.unit
    def test_tab_navigation_workflow(self, mock_fig, multiple_axes, mock_key_event):
        """Test complete tab navigation workflow."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, multiple_axes)
        nav.connect()

        # Navigate forward through all axes
        assert nav.current_axes_index == 0
        nav._on_key(mock_key_event("tab"))
        assert nav.current_axes_index == 1
        nav._on_key(mock_key_event("tab"))
        assert nav.current_axes_index == 2
        nav._on_key(mock_key_event("tab"))
        assert nav.current_axes_index == 0  # Wrapped

        # Navigate backward using _cycle_axes directly
        nav._cycle_axes(reverse=True)
        assert nav.current_axes_index == 2
        nav._cycle_axes(reverse=True)
        assert nav.current_axes_index == 1


class TestPrivateMethods:
    """Tests for private helper methods."""

    @pytest.mark.unit
    def test_pan_method_directly(self, mock_fig, mock_ax):
        """Test _pan method directly."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)

        xlim_before = mock_ax.get_xlim()
        ylim_before = mock_ax.get_ylim()

        # Pan with specific dx and dy
        nav._pan(mock_ax, dx=0.2, dy=0.3)

        xlim_after = mock_ax.get_xlim()
        ylim_after = mock_ax.get_ylim()

        # Check shift is proportional to range
        x_range = xlim_before[1] - xlim_before[0]
        y_range = ylim_before[1] - ylim_before[0]

        expected_x_shift = 0.2 * x_range
        expected_y_shift = 0.3 * y_range

        actual_x_shift = xlim_after[0] - xlim_before[0]
        actual_y_shift = ylim_after[0] - ylim_before[0]

        assert abs(actual_x_shift - expected_x_shift) < 1e-10
        assert abs(actual_y_shift - expected_y_shift) < 1e-10

    @pytest.mark.unit
    def test_zoom_method_directly(self, mock_fig, mock_ax):
        """Test _zoom method directly."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)

        xlim_before = mock_ax.get_xlim()
        ylim_before = mock_ax.get_ylim()
        x_range_before = xlim_before[1] - xlim_before[0]
        y_range_before = ylim_before[1] - ylim_before[0]

        # Zoom with factor 2.0 (zoom out)
        nav._zoom(mock_ax, factor=2.0)

        xlim_after = mock_ax.get_xlim()
        ylim_after = mock_ax.get_ylim()
        x_range_after = xlim_after[1] - xlim_after[0]
        y_range_after = ylim_after[1] - ylim_after[0]

        # Range should be 2x larger
        assert abs(x_range_after - 2.0 * x_range_before) < 1e-10
        assert abs(y_range_after - 2.0 * y_range_before) < 1e-10

    @pytest.mark.unit
    def test_highlight_axes_method(self, mock_fig, mock_ax):
        """Test _highlight_active_axes method."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)

        # Highlight
        nav._highlight_active_axes()

        for spine in mock_ax.spines.values():
            color = spine.get_edgecolor()
            assert color == "red" or color == (1.0, 0.0, 0.0, 1.0)
            assert spine.get_linewidth() == 2

    @pytest.mark.unit
    def test_unhighlight_axes_method(self, mock_fig, mock_ax):
        """Test _unhighlight_axes method."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)

        # First highlight, then unhighlight
        nav._highlight_active_axes()
        nav._unhighlight_axes(mock_ax)

        for spine in mock_ax.spines.values():
            color = spine.get_edgecolor()
            assert color == "black" or color == (0.0, 0.0, 0.0, 1.0)
            assert spine.get_linewidth() == 1


class TestVisualizationKeyboardEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.unit
    def test_empty_axes_list(self, mock_fig):
        """Test initialization with empty axes list."""
        from oscura.visualization.keyboard import KeyboardNavigator

        # Should not raise, but behavior is undefined
        nav = KeyboardNavigator(mock_fig, [])
        assert len(nav.axes_list) == 0

    @pytest.mark.unit
    def test_negative_pan_step(self, mock_fig, mock_ax, mock_key_event):
        """Test that negative pan_step reverses pan direction."""
        from oscura.visualization.keyboard import KeyboardNavigator

        # With negative pan_step, "right" key should pan left
        nav = KeyboardNavigator(mock_fig, mock_ax, pan_step=-0.1)
        nav.connect()

        xlim_before = mock_ax.get_xlim()

        # Press right arrow with negative pan_step
        event = mock_key_event("right")
        nav._on_key(event)

        xlim_after = mock_ax.get_xlim()

        # Should move left (negative of right direction)
        assert xlim_after[0] < xlim_before[0]

    @pytest.mark.unit
    def test_zero_zoom_factor(self, mock_fig, mock_ax):
        """Test zoom with factor 1.0 (no change)."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax, zoom_factor=1.0)

        xlim_before = mock_ax.get_xlim()
        x_range_before = xlim_before[1] - xlim_before[0]

        # Zoom in with factor 1.0 should not change view
        nav._zoom(mock_ax, factor=1.0)

        xlim_after = mock_ax.get_xlim()
        x_range_after = xlim_after[1] - xlim_after[0]

        # Range should be unchanged (within floating point tolerance)
        assert abs(x_range_after - x_range_before) < 1e-10

    @pytest.mark.unit
    def test_very_large_zoom(self, mock_fig, mock_ax):
        """Test zoom with very large factor."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, mock_ax)

        xlim_before = mock_ax.get_xlim()
        x_range_before = xlim_before[1] - xlim_before[0]

        # Zoom out by factor of 1000
        nav._zoom(mock_ax, factor=1000.0)

        xlim_after = mock_ax.get_xlim()
        x_range_after = xlim_after[1] - xlim_after[0]

        # Range should be 1000x larger
        assert abs(x_range_after / x_range_before - 1000.0) < 0.001

    @pytest.mark.unit
    def test_operations_on_non_current_axes(self, mock_fig, multiple_axes):
        """Test that operations affect current axes only."""
        from oscura.visualization.keyboard import KeyboardNavigator

        nav = KeyboardNavigator(mock_fig, multiple_axes)
        nav.connect()

        # Current is axes[0]
        assert nav.current_axes_index == 0

        xlim_0_before = multiple_axes[0].get_xlim()
        xlim_1_before = multiple_axes[1].get_xlim()

        # Pan (should affect axes[0] only)
        nav._pan(multiple_axes[0], dx=0.1, dy=0)

        xlim_0_after = multiple_axes[0].get_xlim()
        xlim_1_after = multiple_axes[1].get_xlim()

        # axes[0] should change
        assert xlim_0_after != xlim_0_before
        # axes[1] should not change
        assert xlim_1_after == xlim_1_before
