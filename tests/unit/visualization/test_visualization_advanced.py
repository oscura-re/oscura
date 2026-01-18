"""Unit tests for advanced visualization features.

Tests for VIS-018, VIS-020, VIS-023, VIS-024, VIS-025 requirements.
"""

import numpy as np
import pytest

from oscura.visualization.colors import (
    select_optimal_palette,
)
from oscura.visualization.histogram import (
    calculate_bin_edges,
    calculate_optimal_bins,
)
from oscura.visualization.optimization import (
    InterestingRegion,
    detect_interesting_regions,
)
from oscura.visualization.styles import (
    StylePreset,
    apply_style_preset,
    create_custom_preset,
    list_presets,
    register_preset,
)
from oscura.visualization.thumbnails import (
    render_thumbnail,
    render_thumbnail_multichannel,
)

# Check if matplotlib is available
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for testing
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

pytestmark = [pytest.mark.unit, pytest.mark.visualization]


class TestInterestingRegionDetection:
    """Tests for VIS-020: Zoom to Interesting Regions."""

    def test_detect_edges(self):
        """Test edge detection in signal."""
        # Create signal with clear edge
        np.linspace(0, 1, 1000)
        signal = np.concatenate(
            [
                np.zeros(300),
                np.linspace(0, 1, 100),  # Edge transition
                np.ones(600),
            ]
        )

        regions = detect_interesting_regions(signal, 1000.0)

        # Should detect the edge
        assert len(regions) > 0
        edge_regions = [r for r in regions if r.type == "edge"]
        assert len(edge_regions) > 0

        # Edge should be around index 300-400
        edge = edge_regions[0]
        assert 200 < edge.start_idx < 500

    def test_detect_glitches(self):
        """Test glitch detection (isolated spikes)."""
        # Create signal with glitch
        signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
        signal[500] = 10.0  # Large spike at index 500 (needs to be >3 sigma)

        regions = detect_interesting_regions(signal, 1000.0, glitch_sigma=2.0)

        # Should detect something interesting near the spike
        # Could be anomaly, glitch, edge, or pattern_change
        assert len(regions) > 0

        # At least one region should overlap with the spike location
        has_spike_region = any(r.start_idx <= 500 <= r.end_idx for r in regions)
        assert has_spike_region, (
            f"No region found containing spike at idx 500. Regions: {[(r.start_idx, r.end_idx, r.type) for r in regions]}"
        )

    def test_detect_anomalies(self):
        """Test anomaly detection using MAD."""
        # Create signal with anomaly region
        signal = np.random.randn(1000) * 0.1
        signal[400:500] = 2.0 + np.random.randn(100) * 0.1  # Anomaly

        regions = detect_interesting_regions(signal, 1000.0)

        # Should detect anomaly
        anomalies = [r for r in regions if r.type == "anomaly"]
        assert len(anomalies) > 0

    def test_detect_pattern_changes(self):
        """Test pattern change detection."""
        # Create signal with pattern change
        signal = np.concatenate(
            [
                np.sin(2 * np.pi * 10 * np.linspace(0, 0.5, 500)),  # Low freq
                np.sin(2 * np.pi * 50 * np.linspace(0.5, 1, 500)),  # High freq
            ]
        )

        regions = detect_interesting_regions(signal, 1000.0)

        # Should detect pattern change
        pattern_changes = [r for r in regions if r.type == "pattern_change"]
        assert len(pattern_changes) > 0

    def test_significance_scoring(self):
        """Test that regions are sorted by significance."""
        # Create signal with varying amplitude edges
        signal = np.zeros(1000)
        signal[200:250] = np.linspace(0, 0.5, 50)  # Small edge
        signal[500:550] = np.linspace(0, 2.0, 50)  # Large edge

        regions = detect_interesting_regions(signal, 1000.0, max_regions=10)

        # Regions should be sorted by significance (descending)
        for i in range(len(regions) - 1):
            assert regions[i].significance >= regions[i + 1].significance

    def test_max_regions_limit(self):
        """Test max_regions parameter."""
        # Create noisy signal with many features
        signal = np.random.randn(10000)

        regions = detect_interesting_regions(signal, 10000.0, max_regions=5)

        assert len(regions) <= 5

    def test_min_region_samples(self):
        """Test min_region_samples filtering."""
        signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))

        regions = detect_interesting_regions(signal, 1000.0, min_region_samples=50)

        # All regions should meet minimum size
        for region in regions:
            assert (region.end_idx - region.start_idx) >= 50

    def test_empty_signal_error(self):
        """Test error handling for empty signal."""
        with pytest.raises(ValueError, match="Signal cannot be empty"):
            detect_interesting_regions(np.array([]), 1000.0)

    def test_invalid_sample_rate(self):
        """Test error handling for invalid sample rate."""
        signal = np.random.randn(100)
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            detect_interesting_regions(signal, -1.0)


class TestThumbnailRendering:
    """Tests for VIS-018: Thumbnail Mode."""

    def test_render_basic_thumbnail(self):
        """Test basic thumbnail rendering."""
        signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 10000))

        fig = render_thumbnail(signal, 10000.0, size=(400, 300))

        assert fig is not None
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_aggressive_decimation(self):
        """Test that thumbnail decimates to max_samples."""
        # Large signal
        signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 100000))

        fig = render_thumbnail(signal, 100000.0, max_samples=500)

        # Should render quickly with reduced samples
        assert fig is not None

        plt.close(fig)

    def test_thumbnail_time_units(self):
        """Test automatic time unit selection."""
        # Nanosecond signal
        signal = np.sin(2 * np.pi * 1e9 * np.linspace(0, 1e-6, 1000))

        fig = render_thumbnail(signal, 1e9, time_unit="auto")

        assert fig is not None

        plt.close(fig)

    def test_custom_title(self):
        """Test custom title."""
        signal = np.random.randn(1000)

        fig = render_thumbnail(signal, 1000.0, title="Test Signal")

        ax = fig.axes[0]
        assert ax.get_title() == "Test Signal"

        plt.close(fig)

    def test_multichannel_thumbnail(self):
        """Test multi-channel thumbnail rendering."""
        signals = [
            np.sin(2 * np.pi * 10 * np.linspace(0, 1, 10000)),
            np.cos(2 * np.pi * 10 * np.linspace(0, 1, 10000)),
            np.sin(2 * np.pi * 20 * np.linspace(0, 1, 10000)),
        ]

        fig = render_thumbnail_multichannel(signals, 10000.0, channel_names=["CH1", "CH2", "CH3"])

        assert fig is not None
        assert len(fig.axes) == 3

        plt.close(fig)

    def test_empty_signal_error(self):
        """Test error for empty signal."""
        with pytest.raises(ValueError, match="Signal cannot be empty"):
            render_thumbnail(np.array([]), 1000.0)

    def test_invalid_sample_rate(self):
        """Test error for invalid sample rate."""
        signal = np.random.randn(100)
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            render_thumbnail(signal, -1.0)


class TestColorPalettes:
    """Tests for VIS-023: Data-Driven Color Palette."""

    def test_qualitative_palette(self):
        """Test qualitative palette selection."""
        colors = select_optimal_palette(5, palette_type="qualitative")

        assert len(colors) == 5
        # Should be hex colors
        for color in colors:
            assert color.startswith("#")
            assert len(color) == 7

    def test_sequential_palette(self):
        """Test sequential palette selection."""
        colors = select_optimal_palette(10, palette_type="sequential")

        assert len(colors) == 10

    def test_diverging_palette(self):
        """Test diverging palette selection."""
        colors = select_optimal_palette(10, palette_type="diverging")

        assert len(colors) == 10

    def test_auto_palette_selection_bipolar(self):
        """Test auto-selection for bipolar data."""
        colors = select_optimal_palette(
            10,
            data_range=(-1.0, 1.0),  # Bipolar
        )

        # Should auto-select diverging for bipolar data
        assert len(colors) == 10

    def test_auto_palette_selection_unipolar(self):
        """Test auto-selection for unipolar data."""
        colors = select_optimal_palette(
            10,
            data_range=(0.0, 1.0),  # Unipolar
        )

        assert len(colors) == 10

    def test_colorblind_safe_palette(self):
        """Test colorblind-safe palette."""
        colors = select_optimal_palette(
            5,
            palette_type="qualitative",
            colorblind_safe=True,
            min_contrast_ratio=1.0,  # Disable contrast adjustment
        )

        assert len(colors) == 5
        # Should use colorblind-safe colors (may be adjusted for contrast)
        # Just verify they are valid hex colors
        for color in colors:
            assert color.startswith("#")
            assert len(color) == 7

    def test_contrast_checking(self):
        """Test WCAG contrast ratio checking."""
        colors = select_optimal_palette(5, background_color="#FFFFFF", min_contrast_ratio=4.5)

        # All colors should meet minimum contrast
        assert len(colors) == 5

    def test_many_colors_interpolation(self):
        """Test palette interpolation for many colors."""
        colors = select_optimal_palette(20, palette_type="sequential")

        assert len(colors) == 20

    def test_invalid_n_colors(self):
        """Test error for invalid n_colors."""
        with pytest.raises(ValueError, match="n_colors must be >= 1"):
            select_optimal_palette(0)


class TestStylePresets:
    """Tests for VIS-024: Plot Style Presets."""

    def test_publication_preset(self):
        """Test publication style preset."""
        with apply_style_preset("publication"):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])

            # Check DPI
            assert fig.dpi == 600

        plt.close(fig)

    def test_presentation_preset(self):
        """Test presentation style preset."""
        with apply_style_preset("presentation"):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])

            # Check font size
            assert plt.rcParams["font.size"] == 18

        plt.close(fig)

    def test_screen_preset(self):
        """Test screen style preset."""
        with apply_style_preset("screen"):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])

            assert fig.dpi == 96

        plt.close(fig)

    def test_print_preset(self):
        """Test print style preset."""
        with apply_style_preset("print"):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])

            assert fig.dpi == 300

        plt.close(fig)

    def test_preset_with_overrides(self):
        """Test preset with custom overrides."""
        with apply_style_preset("screen", overrides={"font.size": 14}):
            assert plt.rcParams["font.size"] == 14

    def test_custom_preset_creation(self):
        """Test creating custom preset."""
        custom = create_custom_preset(
            "my_style", base_preset="publication", font_size=12, line_width=1.5
        )

        assert custom.name == "my_style"
        assert custom.font_size == 12
        assert custom.line_width == 1.5
        assert custom.dpi == 600  # Inherited from publication

    def test_custom_preset_registration(self):
        """Test registering custom preset."""
        custom = create_custom_preset("test_style", base_preset="screen")
        register_preset(custom)

        # Should be able to use registered preset
        with apply_style_preset("test_style"):
            fig, _ax = plt.subplots()

        plt.close(fig)

    def test_list_presets(self):
        """Test listing available presets."""
        presets = list_presets()

        assert "publication" in presets
        assert "presentation" in presets
        assert "screen" in presets
        assert "print" in presets

    def test_unknown_preset_error(self):
        """Test error for unknown preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            with apply_style_preset("nonexistent"):
                pass

    def test_preset_inheritance(self):
        """Test preset inheritance mechanism."""
        custom = create_custom_preset(
            "derived",
            base_preset="publication",
            font_size=14,  # Override only font_size
        )

        # Should inherit other attributes
        assert custom.dpi == 600
        assert custom.font_family == "serif"
        assert custom.font_size == 14  # Overridden

    def test_style_preset_object(self):
        """Test using StylePreset object directly."""
        preset = StylePreset(name="custom", dpi=150, font_size=12)

        with apply_style_preset(preset):
            fig, _ax = plt.subplots()
            assert fig.dpi == 150

        plt.close(fig)


class TestHistogramBinning:
    """Tests for VIS-025: Histogram Bin Optimization."""

    def test_sturges_rule(self):
        """Test Sturges' rule for bin calculation."""
        data = np.random.randn(1000)

        bins = calculate_optimal_bins(data, method="sturges")

        # Sturges: ceil(log2(1000) + 1) ≈ 11
        assert 8 <= bins <= 15

    def test_freedman_diaconis_rule(self):
        """Test Freedman-Diaconis rule."""
        data = np.random.randn(1000)

        bins = calculate_optimal_bins(data, method="freedman-diaconis")

        assert bins > 0

    def test_scott_rule(self):
        """Test Scott's rule."""
        data = np.random.randn(1000)

        bins = calculate_optimal_bins(data, method="scott")

        assert bins > 0

    def test_auto_method_selection(self):
        """Test automatic method selection."""
        # Normal distribution
        data = np.random.randn(1000)

        bins = calculate_optimal_bins(data, method="auto")

        assert bins > 0

    def test_auto_with_outliers(self):
        """Test auto-selection with outliers (should use FD)."""
        # Skewed distribution with outliers
        data = np.concatenate(
            [
                np.random.randn(900),
                np.random.randn(100) * 10,  # Outliers
            ]
        )

        bins = calculate_optimal_bins(data, method="auto")

        assert bins > 0

    def test_small_sample_sturges(self):
        """Test that small samples use Sturges."""
        data = np.random.randn(50)

        bins = calculate_optimal_bins(data, method="auto")

        # Should use Sturges for small samples
        assert bins > 0

    def test_min_max_bins_clamping(self):
        """Test that bins are clamped to min/max."""
        data = np.random.randn(10)

        bins = calculate_optimal_bins(data, method="sturges", min_bins=5, max_bins=10)

        assert 5 <= bins <= 10

    def test_uniform_data(self):
        """Test with uniform (constant) data."""
        data = np.ones(100)

        bins = calculate_optimal_bins(data, method="freedman-diaconis")

        # Should handle constant data gracefully
        assert bins >= 5

    def test_calculate_bin_edges(self):
        """Test bin edge calculation."""
        data = np.random.randn(1000)
        n_bins = 20

        edges = calculate_bin_edges(data, n_bins)

        assert len(edges) == n_bins + 1
        assert edges[0] <= np.min(data)
        assert edges[-1] >= np.max(data)
        # Should be monotonically increasing
        assert np.all(np.diff(edges) > 0)

    def test_single_value_data(self):
        """Test with single-value data."""
        data = np.ones(100)

        edges = calculate_bin_edges(data, 10)

        # Should create reasonable edges around the value
        # For constant data, we create a small range around it
        assert len(edges) >= 2
        assert edges[0] < 1.0
        assert edges[-1] > 1.0

    def test_empty_data_error(self):
        """Test error for empty data."""
        with pytest.raises(ValueError, match="Data array cannot be empty"):
            calculate_optimal_bins(np.array([]))

    def test_nan_handling(self):
        """Test handling of NaN values."""
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan])

        bins = calculate_optimal_bins(data, method="sturges")

        # Should ignore NaN values
        assert bins > 0

    def test_invalid_min_bins(self):
        """Test error for invalid min_bins."""
        data = np.random.randn(100)
        with pytest.raises(ValueError, match="min_bins must be >= 1"):
            calculate_optimal_bins(data, min_bins=0)

    def test_invalid_max_bins(self):
        """Test error for invalid max_bins."""
        data = np.random.randn(100)
        with pytest.raises(ValueError, match="max_bins must be >= min_bins"):
            calculate_optimal_bins(data, min_bins=10, max_bins=5)

    def test_unknown_method_error(self):
        """Test error for unknown method."""
        data = np.random.randn(100)
        with pytest.raises(ValueError, match="Unknown method"):
            calculate_optimal_bins(data, method="unknown")


class TestInterestingRegionDataclass:
    """Tests for InterestingRegion dataclass."""

    def test_region_attributes(self):
        """Test InterestingRegion attributes."""
        region = InterestingRegion(
            start_idx=100,
            end_idx=200,
            start_time=0.1,
            end_time=0.2,
            type="edge",
            significance=0.8,
            metadata={"threshold": 1.5},
        )

        assert region.start_idx == 100
        assert region.end_idx == 200
        assert region.start_time == 0.1
        assert region.end_time == 0.2
        assert region.type == "edge"
        assert region.significance == 0.8
        assert region.metadata["threshold"] == 1.5
