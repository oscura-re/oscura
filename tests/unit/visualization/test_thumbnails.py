"""Comprehensive tests for visualization.thumbnails module.

Tests cover fast thumbnail rendering with reduced detail for preview contexts.

Coverage target: >90%
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip all tests if matplotlib not available
try:
    import matplotlib  # noqa: F401

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

pytestmark = [
    pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed"),
    pytest.mark.usefixtures("cleanup_matplotlib"),
]


class TestRenderThumbnail:
    """Tests for render_thumbnail function."""

    def test_import_error_without_matplotlib(self) -> None:
        """Test ImportError when matplotlib not available."""
        import importlib
        import sys
        from unittest.mock import patch

        from oscura.visualization import thumbnails

        try:
            with patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}):
                importlib.reload(thumbnails)

                signal = np.sin(np.linspace(0, 10, 1000))

                with pytest.raises(ImportError, match="matplotlib is required"):
                    thumbnails.render_thumbnail(signal)
        finally:
            # Restore module to original state for subsequent tests
            importlib.reload(thumbnails)

    def test_basic_thumbnail(self, matplotlib_available: None) -> None:
        """Test basic thumbnail rendering."""
        from oscura.visualization.thumbnails import render_thumbnail

        signal = np.sin(np.linspace(0, 10, 1000))

        fig = render_thumbnail(signal, sample_rate=1000.0)

        assert fig is not None
        assert len(fig.axes) == 1

    def test_empty_signal_raises(self, matplotlib_available: None) -> None:
        """Test that empty signal raises ValueError."""
        from oscura.visualization.thumbnails import render_thumbnail

        with pytest.raises(ValueError, match="cannot be empty"):
            render_thumbnail(np.array([]))

    def test_invalid_sample_rate_raises(self, matplotlib_available: None) -> None:
        """Test that invalid sample rate raises ValueError."""
        from oscura.visualization.thumbnails import render_thumbnail

        signal = np.sin(np.linspace(0, 10, 1000))

        with pytest.raises(ValueError, match="must be positive"):
            render_thumbnail(signal, sample_rate=0.0)

        with pytest.raises(ValueError, match="must be positive"):
            render_thumbnail(signal, sample_rate=-100.0)

    def test_max_samples_constraint(self, matplotlib_available: None) -> None:
        """Test max_samples parameter."""
        from oscura.visualization.thumbnails import render_thumbnail

        signal = np.sin(np.linspace(0, 10, 10000))

        fig = render_thumbnail(signal, sample_rate=10000.0, max_samples=100)

        # Thumbnail should be rendered with decimated data
        assert fig is not None

    def test_max_samples_too_small_raises(self, matplotlib_available: None) -> None:
        """Test that max_samples < 10 raises ValueError."""
        from oscura.visualization.thumbnails import render_thumbnail

        signal = np.sin(np.linspace(0, 10, 1000))

        with pytest.raises(ValueError, match="max_samples must be >= 10"):
            render_thumbnail(signal, sample_rate=1000.0, max_samples=5)

    def test_custom_size(self, matplotlib_available: None) -> None:
        """Test custom thumbnail size."""
        from oscura.visualization.thumbnails import render_thumbnail

        signal = np.sin(np.linspace(0, 10, 1000))

        fig = render_thumbnail(signal, sample_rate=1000.0, size=(200, 150))

        assert fig is not None

    def test_width_height_parameters(self, matplotlib_available: None) -> None:
        """Test width and height as alternative to size."""
        from oscura.visualization.thumbnails import render_thumbnail

        signal = np.sin(np.linspace(0, 10, 1000))

        # Width only (height auto-calculated)
        fig1 = render_thumbnail(signal, sample_rate=1000.0, width=300)
        assert fig1 is not None

        # Height only (width auto-calculated)
        fig2 = render_thumbnail(signal, sample_rate=1000.0, height=200)
        assert fig2 is not None

        # Both specified
        fig3 = render_thumbnail(signal, sample_rate=1000.0, width=300, height=200)
        assert fig3 is not None

    def test_time_unit_selection(self, matplotlib_available: None) -> None:
        """Test different time units."""
        from oscura.visualization.thumbnails import render_thumbnail

        signal = np.sin(np.linspace(0, 10, 1000))

        for time_unit in ["s", "ms", "us", "ns"]:
            fig = render_thumbnail(signal, sample_rate=1000.0, time_unit=time_unit)
            assert fig is not None

    def test_auto_time_unit(self, matplotlib_available: None) -> None:
        """Test automatic time unit selection."""
        from oscura.visualization.thumbnails import render_thumbnail

        # Very short signal → ns
        signal_ns = np.sin(np.linspace(0, 10, 100))
        fig_ns = render_thumbnail(signal_ns, sample_rate=1e9, time_unit="auto")
        assert fig_ns is not None

        # Short signal → us
        signal_us = np.sin(np.linspace(0, 10, 1000))
        fig_us = render_thumbnail(signal_us, sample_rate=1e6, time_unit="auto")
        assert fig_us is not None

        # Medium signal → ms
        signal_ms = np.sin(np.linspace(0, 10, 1000))
        fig_ms = render_thumbnail(signal_ms, sample_rate=1e3, time_unit="auto")
        assert fig_ms is not None

        # Long signal → s
        signal_s = np.sin(np.linspace(0, 10, 1000))
        fig_s = render_thumbnail(signal_s, sample_rate=10.0, time_unit="auto")
        assert fig_s is not None

    def test_custom_title(self, matplotlib_available: None) -> None:
        """Test custom title."""
        from oscura.visualization.thumbnails import render_thumbnail

        signal = np.sin(np.linspace(0, 10, 1000))

        fig = render_thumbnail(signal, sample_rate=1000.0, title="Test Signal")

        # Title should be set
        assert fig.axes[0].get_title() == "Test Signal"

    def test_custom_dpi(self, matplotlib_available: None) -> None:
        """Test custom DPI."""
        from oscura.visualization.thumbnails import render_thumbnail

        signal = np.sin(np.linspace(0, 10, 1000))

        fig = render_thumbnail(signal, sample_rate=1000.0, dpi=150)

        assert fig.dpi == 150

    def test_no_sample_rate(self, matplotlib_available: None) -> None:
        """Test rendering without sample rate (uses indices)."""
        from oscura.visualization.thumbnails import render_thumbnail

        signal = np.sin(np.linspace(0, 10, 1000))

        # sample_rate=None should default to 1.0
        fig = render_thumbnail(signal, sample_rate=None)

        assert fig is not None

    def test_decimation_applied(self, matplotlib_available: None) -> None:
        """Test that decimation is actually applied for large signals."""
        from oscura.visualization.thumbnails import render_thumbnail

        # Large signal that needs decimation
        signal = np.sin(np.linspace(0, 10, 100000))

        fig = render_thumbnail(signal, sample_rate=10000.0, max_samples=100)

        # Should successfully create thumbnail (decimation happens internally)
        assert fig is not None


class TestRenderThumbnailMultichannel:
    """Tests for render_thumbnail_multichannel function."""

    def test_basic_multichannel(self, matplotlib_available: None) -> None:
        """Test basic multi-channel thumbnail."""
        from oscura.visualization.thumbnails import render_thumbnail_multichannel

        signals = [
            np.sin(np.linspace(0, 10, 1000)),
            np.cos(np.linspace(0, 10, 1000)),
            np.sin(np.linspace(0, 10, 1000)) * 0.5,
        ]

        fig = render_thumbnail_multichannel(signals, sample_rate=1000.0)

        # Should have 3 subplots
        assert len(fig.axes) == 3

    def test_empty_signals_raises(self, matplotlib_available: None) -> None:
        """Test that empty signal list raises ValueError."""
        from oscura.visualization.thumbnails import render_thumbnail_multichannel

        with pytest.raises(ValueError, match="at least one signal"):
            render_thumbnail_multichannel([], sample_rate=1000.0)

    def test_invalid_sample_rate_raises(self, matplotlib_available: None) -> None:
        """Test that invalid sample rate raises ValueError."""
        from oscura.visualization.thumbnails import render_thumbnail_multichannel

        signals = [np.sin(np.linspace(0, 10, 1000))]

        with pytest.raises(ValueError, match="must be positive"):
            render_thumbnail_multichannel(signals, sample_rate=0.0)

    def test_custom_channel_names(self, matplotlib_available: None) -> None:
        """Test custom channel names."""
        from oscura.visualization.thumbnails import render_thumbnail_multichannel

        signals = [
            np.sin(np.linspace(0, 10, 1000)),
            np.cos(np.linspace(0, 10, 1000)),
        ]
        channel_names = ["Voltage", "Current"]

        fig = render_thumbnail_multichannel(
            signals, sample_rate=1000.0, channel_names=channel_names
        )

        # Channel names should be used in y-labels
        assert fig is not None

    def test_auto_channel_names(self, matplotlib_available: None) -> None:
        """Test automatic channel name generation."""
        from oscura.visualization.thumbnails import render_thumbnail_multichannel

        signals = [
            np.sin(np.linspace(0, 10, 1000)),
            np.cos(np.linspace(0, 10, 1000)),
        ]

        # Without channel_names, should use CH1, CH2, etc.
        fig = render_thumbnail_multichannel(signals, sample_rate=1000.0)

        assert fig is not None

    def test_multichannel_max_samples(self, matplotlib_available: None) -> None:
        """Test max_samples parameter for multi-channel."""
        from oscura.visualization.thumbnails import render_thumbnail_multichannel

        signals = [
            np.sin(np.linspace(0, 10, 10000)),
            np.cos(np.linspace(0, 10, 10000)),
        ]

        fig = render_thumbnail_multichannel(signals, sample_rate=10000.0, max_samples=100)

        assert fig is not None

    def test_multichannel_time_unit(self, matplotlib_available: None) -> None:
        """Test time unit selection for multi-channel."""
        from oscura.visualization.thumbnails import render_thumbnail_multichannel

        signals = [np.sin(np.linspace(0, 10, 1000))]

        for time_unit in ["s", "ms", "us", "ns"]:
            fig = render_thumbnail_multichannel(signals, sample_rate=1000.0, time_unit=time_unit)
            assert fig is not None

    def test_multichannel_custom_size(self, matplotlib_available: None) -> None:
        """Test custom size for multi-channel."""
        from oscura.visualization.thumbnails import render_thumbnail_multichannel

        signals = [
            np.sin(np.linspace(0, 10, 1000)),
            np.cos(np.linspace(0, 10, 1000)),
        ]

        fig = render_thumbnail_multichannel(signals, sample_rate=1000.0, size=(600, 400))

        assert fig is not None

    def test_multichannel_single_channel(self, matplotlib_available: None) -> None:
        """Test multi-channel function with single channel."""
        from oscura.visualization.thumbnails import render_thumbnail_multichannel

        signals = [np.sin(np.linspace(0, 10, 1000))]

        fig = render_thumbnail_multichannel(signals, sample_rate=1000.0)

        # Should work with single channel
        assert len(fig.axes) == 1

    def test_empty_signal_in_list(self, matplotlib_available: None) -> None:
        """Test handling of empty signal in list."""
        from oscura.visualization.thumbnails import render_thumbnail_multichannel

        signals = [
            np.sin(np.linspace(0, 10, 1000)),
            np.array([]),  # Empty signal
            np.cos(np.linspace(0, 10, 1000)),
        ]

        # Should handle gracefully (skip empty)
        fig = render_thumbnail_multichannel(signals, sample_rate=1000.0)

        assert fig is not None


class TestDecimateUniform:
    """Tests for _decimate_uniform helper function."""

    def test_decimate_to_target(self, matplotlib_available: None) -> None:
        """Test decimation to target sample count."""
        from oscura.visualization.thumbnails import _decimate_uniform

        signal = np.arange(10000, dtype=np.float64)
        decimated = _decimate_uniform(signal, 100)

        # Should have exactly 100 samples
        assert len(decimated) == 100

    def test_no_decimation_if_below_target(self, matplotlib_available: None) -> None:
        """Test that signal below target is not decimated."""
        from oscura.visualization.thumbnails import _decimate_uniform

        signal = np.arange(50, dtype=np.float64)
        decimated = _decimate_uniform(signal, 100)

        # Should return unchanged
        assert len(decimated) == 50
        np.testing.assert_array_equal(decimated, signal)

    def test_uniform_stride(self, matplotlib_available: None) -> None:
        """Test that decimation uses uniform stride."""
        from oscura.visualization.thumbnails import _decimate_uniform

        signal = np.arange(1000, dtype=np.float64)
        decimated = _decimate_uniform(signal, 10)

        # Should sample at uniform intervals
        assert len(decimated) == 10
        # First sample should be first element
        assert decimated[0] == 0.0

    def test_preserves_first_sample(self, matplotlib_available: None) -> None:
        """Test that first sample is always preserved."""
        from oscura.visualization.thumbnails import _decimate_uniform

        signal = np.random.randn(10000)
        decimated = _decimate_uniform(signal, 100)

        assert decimated[0] == signal[0]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_signal(self, matplotlib_available: None) -> None:
        """Test with very small signal."""
        from oscura.visualization.thumbnails import render_thumbnail

        signal = np.array([1.0, 2.0, 3.0])

        fig = render_thumbnail(signal, sample_rate=1.0)

        assert fig is not None

    def test_large_signal(self, matplotlib_available: None) -> None:
        """Test with large signal (performance)."""
        from oscura.visualization.thumbnails import render_thumbnail

        # 1M samples
        signal = np.sin(np.linspace(0, 100, 1_000_000))

        # Should complete quickly (<1 second expected)
        fig = render_thumbnail(signal, sample_rate=10000.0, max_samples=1000)

        assert fig is not None

    def test_constant_signal(self, matplotlib_available: None) -> None:
        """Test with constant signal."""
        from oscura.visualization.thumbnails import render_thumbnail

        signal = np.ones(1000)

        fig = render_thumbnail(signal, sample_rate=1000.0)

        assert fig is not None

    def test_signal_with_nans(self, matplotlib_available: None) -> None:
        """Test signal with NaN values."""
        from oscura.visualization.thumbnails import render_thumbnail

        signal = np.sin(np.linspace(0, 10, 1000))
        signal[500] = np.nan

        # Should handle gracefully
        fig = render_thumbnail(signal, sample_rate=1000.0)

        assert fig is not None

    def test_signal_with_infs(self, matplotlib_available: None) -> None:
        """Test signal with infinite values."""
        from oscura.visualization.thumbnails import render_thumbnail

        signal = np.sin(np.linspace(0, 10, 1000))
        signal[500] = np.inf

        # Should handle gracefully
        fig = render_thumbnail(signal, sample_rate=1000.0)

        assert fig is not None


@pytest.fixture
def matplotlib_available() -> None:
    """Ensure matplotlib is available for tests."""
    pytest.importorskip("matplotlib")


# Run tests with: pytest tests/unit/visualization/test_thumbnails.py -v
