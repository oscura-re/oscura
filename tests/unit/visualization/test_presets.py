"""Comprehensive unit tests for visualization presets.

Tests all preset configurations, loading, and application functionality.


Target: 90%+ coverage
"""

import re
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import pytest

# Set matplotlib backend before importing presets
matplotlib.use("Agg")

from oscura.visualization.presets import (
    DARK_THEME_PRESET,
    IEEE_DOUBLE_COLUMN_PRESET,
    IEEE_PUBLICATION_PRESET,
    PRESENTATION_PRESET,
    PRESETS,
    PRINT_PRESET,
    SCREEN_PRESET,
    VisualizationPreset,
    apply_preset,
    create_custom_preset,
    get_preset_colors,
    list_presets,
)

pytestmark = [pytest.mark.unit, pytest.mark.visualization]


class TestVisualizationPreset:
    """Tests for VisualizationPreset dataclass."""

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_dataclass_initialization(self):
        """Test VisualizationPreset can be instantiated."""
        preset = VisualizationPreset(
            name="test",
            description="Test preset",
            style_params={"font.size": 10},
            color_palette=["#000000", "#FFFFFF"],
        )
        assert preset.name == "test"
        assert preset.description == "Test preset"
        assert preset.dpi == 96  # Default value
        assert preset.figure_size == (10, 6)  # Default value
        assert preset.font_family == "sans-serif"  # Default value
        assert preset.colorblind_safe is True  # Default value
        assert preset.print_optimized is False  # Default value

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_dataclass_custom_values(self):
        """Test VisualizationPreset with all custom values."""
        preset = VisualizationPreset(
            name="custom",
            description="Custom preset",
            style_params={"font.size": 12},
            color_palette=["#FF0000"],
            dpi=300,
            figure_size=(8, 6),
            font_family="serif",
            colorblind_safe=False,
            print_optimized=True,
        )
        assert preset.dpi == 300
        assert preset.figure_size == (8, 6)
        assert preset.font_family == "serif"
        assert preset.colorblind_safe is False
        assert preset.print_optimized is True

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_dataclass_immutability(self):
        """Test that preset attributes can be accessed."""
        preset = IEEE_PUBLICATION_PRESET
        assert preset.name == "ieee_publication"
        assert isinstance(preset.style_params, dict)
        assert isinstance(preset.color_palette, list)


class TestIEEEPublicationPreset:
    """Tests for IEEE_PUBLICATION_PRESET (VIS-020)."""

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_name_and_description(self):
        """Test IEEE publication preset identification."""
        assert IEEE_PUBLICATION_PRESET.name == "ieee_publication"
        assert "IEEE" in IEEE_PUBLICATION_PRESET.description
        assert "single-column" in IEEE_PUBLICATION_PRESET.description

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_dpi_settings(self):
        """Test IEEE preset uses 600 DPI for publication quality."""
        assert IEEE_PUBLICATION_PRESET.dpi == 600
        assert IEEE_PUBLICATION_PRESET.style_params["figure.dpi"] == 600
        assert IEEE_PUBLICATION_PRESET.style_params["savefig.dpi"] == 600

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_figure_size(self):
        """Test IEEE single-column width (3.5 inches)."""
        assert IEEE_PUBLICATION_PRESET.figure_size == (3.5, 2.5)

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_font_settings(self):
        """Test serif fonts and appropriate sizes."""
        assert IEEE_PUBLICATION_PRESET.font_family == "serif"
        assert IEEE_PUBLICATION_PRESET.style_params["font.family"] == "serif"
        assert IEEE_PUBLICATION_PRESET.style_params["font.size"] == 8
        assert IEEE_PUBLICATION_PRESET.style_params["mathtext.fontset"] == "cm"

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_colorblind_safety(self):
        """Test IEEE preset is colorblind-safe."""
        assert IEEE_PUBLICATION_PRESET.colorblind_safe is True

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_print_optimization(self):
        """Test IEEE preset is print-optimized."""
        assert IEEE_PUBLICATION_PRESET.print_optimized is True
        assert IEEE_PUBLICATION_PRESET.style_params["lines.antialiased"] is False
        assert IEEE_PUBLICATION_PRESET.style_params["patch.antialiased"] is False

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_color_palette(self):
        """Test grayscale-friendly color palette."""
        colors = IEEE_PUBLICATION_PRESET.color_palette
        assert len(colors) == 6
        assert colors[0] == "#000000"  # Black
        assert colors[1] == "#555555"  # Dark gray
        assert all(isinstance(c, str) and c.startswith("#") for c in colors)

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_line_and_grid_settings(self):
        """Test line widths appropriate for print."""
        assert IEEE_PUBLICATION_PRESET.style_params["lines.linewidth"] == 0.8
        assert IEEE_PUBLICATION_PRESET.style_params["grid.linewidth"] == 0.4
        assert IEEE_PUBLICATION_PRESET.style_params["grid.linestyle"] == ":"

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_save_format(self):
        """Test PDF save format for publications."""
        assert IEEE_PUBLICATION_PRESET.style_params["savefig.format"] == "pdf"
        assert IEEE_PUBLICATION_PRESET.style_params["savefig.bbox"] == "tight"


class TestIEEEDoubleColumnPreset:
    """Tests for IEEE_DOUBLE_COLUMN_PRESET."""

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_name_and_description(self):
        """Test IEEE double-column preset identification."""
        assert IEEE_DOUBLE_COLUMN_PRESET.name == "ieee_double_column"
        assert "double-column" in IEEE_DOUBLE_COLUMN_PRESET.description

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_figure_size(self):
        """Test IEEE double-column width (7.0 inches)."""
        assert IEEE_DOUBLE_COLUMN_PRESET.figure_size == (7.0, 2.5)

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_inherits_from_single_column(self):
        """Test double-column inherits settings from single-column."""
        assert IEEE_DOUBLE_COLUMN_PRESET.dpi == IEEE_PUBLICATION_PRESET.dpi
        assert IEEE_DOUBLE_COLUMN_PRESET.color_palette == IEEE_PUBLICATION_PRESET.color_palette
        assert (
            IEEE_DOUBLE_COLUMN_PRESET.style_params["font.size"]
            == IEEE_PUBLICATION_PRESET.style_params["font.size"]
        )


class TestPresentationPreset:
    """Tests for PRESENTATION_PRESET."""

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_name_and_description(self):
        """Test presentation preset identification."""
        assert PRESENTATION_PRESET.name == "presentation"
        assert "slides" in PRESENTATION_PRESET.description.lower()

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_large_fonts(self):
        """Test large fonts for visibility."""
        assert PRESENTATION_PRESET.style_params["font.size"] == 18
        assert PRESENTATION_PRESET.style_params["axes.titlesize"] == 22
        assert PRESENTATION_PRESET.style_params["legend.fontsize"] == 16

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_bold_lines(self):
        """Test thick lines for visibility."""
        assert PRESENTATION_PRESET.style_params["lines.linewidth"] == 3.0
        assert PRESENTATION_PRESET.style_params["axes.linewidth"] == 2.0

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_screen_dpi(self):
        """Test standard screen DPI."""
        assert PRESENTATION_PRESET.dpi == 96

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_large_figure_size(self):
        """Test large figure size for presentations."""
        assert PRESENTATION_PRESET.figure_size == (12, 7)

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_antialiasing(self):
        """Test antialiasing enabled for screen viewing."""
        assert PRESENTATION_PRESET.style_params["lines.antialiased"] is True
        assert PRESENTATION_PRESET.style_params["patch.antialiased"] is True

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_vibrant_colors(self):
        """Test high-contrast color palette."""
        colors = PRESENTATION_PRESET.color_palette
        assert len(colors) == 6
        assert "#0173B2" in colors  # Blue
        assert "#DE8F05" in colors  # Orange


class TestScreenPreset:
    """Tests for SCREEN_PRESET."""

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_name_and_description(self):
        """Test screen preset identification."""
        assert SCREEN_PRESET.name == "screen"
        assert "screen" in SCREEN_PRESET.description.lower()

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_medium_fonts(self):
        """Test medium-sized fonts for screen viewing."""
        assert SCREEN_PRESET.style_params["font.size"] == 10
        assert SCREEN_PRESET.style_params["axes.titlesize"] == 12

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_standard_figure_size(self):
        """Test standard figure size."""
        assert SCREEN_PRESET.figure_size == (10, 6)

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_vibrant_color_palette(self):
        """Test vibrant colors for screen."""
        colors = SCREEN_PRESET.color_palette
        assert len(colors) == 8
        assert colors[0] == "#1F77B4"  # Default matplotlib blue

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_not_print_optimized(self):
        """Test screen preset is not print-optimized."""
        assert SCREEN_PRESET.print_optimized is False


class TestPrintPreset:
    """Tests for PRINT_PRESET."""

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_name_and_description(self):
        """Test print preset identification."""
        assert PRINT_PRESET.name == "print"
        assert "print" in PRINT_PRESET.description.lower()

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_print_dpi(self):
        """Test 300 DPI for print quality."""
        assert PRINT_PRESET.dpi == 300
        assert PRINT_PRESET.style_params["figure.dpi"] == 300
        assert PRINT_PRESET.style_params["savefig.dpi"] == 300

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_cmyk_safe_colors(self):
        """Test CMYK-safe color palette."""
        colors = PRINT_PRESET.color_palette
        assert len(colors) == 5
        # Should contain CMYK-safe colors
        assert all(c.startswith("#") for c in colors)

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_print_optimization(self):
        """Test print optimization settings."""
        assert PRINT_PRESET.print_optimized is True
        assert PRINT_PRESET.style_params["lines.antialiased"] is False
        assert PRINT_PRESET.style_params["savefig.format"] == "pdf"

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_serif_fonts(self):
        """Test serif fonts for print."""
        assert PRINT_PRESET.font_family == "serif"
        assert PRINT_PRESET.style_params["font.family"] == "serif"


class TestDarkThemePreset:
    """Tests for DARK_THEME_PRESET."""

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_name_and_description(self):
        """Test dark theme preset identification."""
        assert DARK_THEME_PRESET.name == "dark"
        assert "dark" in DARK_THEME_PRESET.description.lower()

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_dark_background(self):
        """Test dark background colors."""
        assert DARK_THEME_PRESET.style_params["figure.facecolor"] == "#1E1E1E"
        assert DARK_THEME_PRESET.style_params["axes.facecolor"] == "#2D2D2D"

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_light_text(self):
        """Test light text colors for dark background."""
        assert DARK_THEME_PRESET.style_params["text.color"] == "#CCCCCC"
        assert DARK_THEME_PRESET.style_params["axes.labelcolor"] == "#CCCCCC"
        assert DARK_THEME_PRESET.style_params["xtick.color"] == "#CCCCCC"

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_high_contrast_colors(self):
        """Test high-contrast color palette for dark theme."""
        colors = DARK_THEME_PRESET.color_palette
        assert len(colors) == 6
        assert "#56B4E9" in colors  # Light blue
        assert "#E69F00" in colors  # Orange


class TestPresetsRegistry:
    """Tests for PRESETS registry."""

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_all_presets_registered(self):
        """Test all presets are in registry."""
        assert "ieee_publication" in PRESETS
        assert "ieee_double_column" in PRESETS
        assert "presentation" in PRESETS
        assert "screen" in PRESETS
        assert "print" in PRESETS
        assert "dark" in PRESETS

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_registry_values_correct(self):
        """Test registry maps to correct preset objects."""
        assert PRESETS["ieee_publication"] is IEEE_PUBLICATION_PRESET
        assert PRESETS["ieee_double_column"] is IEEE_DOUBLE_COLUMN_PRESET
        assert PRESETS["presentation"] is PRESENTATION_PRESET
        assert PRESETS["screen"] is SCREEN_PRESET
        assert PRESETS["print"] is PRINT_PRESET
        assert PRESETS["dark"] is DARK_THEME_PRESET

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_registry_keys_are_strings(self):
        """Test registry keys are all strings."""
        assert all(isinstance(k, str) for k in PRESETS)

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_registry_values_are_presets(self):
        """Test registry values are VisualizationPreset instances."""
        assert all(isinstance(v, VisualizationPreset) for v in PRESETS.values())


class TestApplyPreset:
    """Tests for apply_preset context manager (VIS-020, VIS-024)."""

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_apply_preset_by_name(self):
        """Test applying preset by string name."""
        with apply_preset("screen") as preset:
            assert preset is SCREEN_PRESET
            assert isinstance(preset, VisualizationPreset)

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_apply_preset_by_object(self):
        """Test applying preset by object."""
        with apply_preset(IEEE_PUBLICATION_PRESET) as preset:
            assert preset is IEEE_PUBLICATION_PRESET

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_invalid_preset_name(self):
        """Test ValueError for unknown preset name."""
        with pytest.raises(ValueError, match="Unknown preset"):
            with apply_preset("nonexistent"):
                pass

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_error_includes_available_presets(self):
        """Test error message lists available presets."""
        with pytest.raises(ValueError, match="ieee_publication"):
            with apply_preset("invalid"):
                pass

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_applies_style_params(self):
        """Test that style params are applied to matplotlib."""
        original_size = plt.rcParams["font.size"]

        with apply_preset("presentation"):
            # Should have presentation font size
            assert plt.rcParams["font.size"] == 18

        # Should restore original after context
        assert plt.rcParams["font.size"] == original_size

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_context_restoration(self):
        """Test rcParams are restored after context."""
        original_linewidth = plt.rcParams["lines.linewidth"]

        with apply_preset("ieee_publication"):
            assert plt.rcParams["lines.linewidth"] == 0.8

        assert plt.rcParams["lines.linewidth"] == original_linewidth

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_with_overrides(self):
        """Test applying preset with custom overrides."""
        with apply_preset("screen", overrides={"font.size": 14}) as preset:
            assert plt.rcParams["font.size"] == 14
            # Other settings should still apply
            assert plt.rcParams["axes.titlesize"] == 12

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_overrides_do_not_modify_preset(self):
        """Test overrides don't modify the original preset."""
        original_params = SCREEN_PRESET.style_params.copy()

        with apply_preset("screen", overrides={"font.size": 99}):
            pass

        assert SCREEN_PRESET.style_params == original_params

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_yields_preset_object(self):
        """Test context manager yields preset for color access."""
        with apply_preset("ieee_publication") as preset:
            colors = preset.color_palette
            assert len(colors) == 6
            assert colors[0] == "#000000"

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_multiple_nested_presets(self):
        """Test nested preset contexts."""
        with apply_preset("screen"):
            screen_size = plt.rcParams["font.size"]

            with apply_preset("presentation"):
                assert plt.rcParams["font.size"] == 18

            # Should restore to screen settings
            assert plt.rcParams["font.size"] == screen_size

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_matplotlib_not_available(self):
        """Test ImportError when matplotlib not available."""
        with patch("oscura.visualization.presets.HAS_MATPLOTLIB", False):
            with pytest.raises(ImportError, match="matplotlib is required"):
                with apply_preset("screen"):
                    pass


class TestGetPresetColors:
    """Tests for get_preset_colors function (VIS-023)."""

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_get_colors_by_name(self):
        """Test getting colors by preset name."""
        colors = get_preset_colors("screen")
        assert isinstance(colors, list)
        assert len(colors) == 8
        assert colors[0] == "#1F77B4"

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_get_colors_by_object(self):
        """Test getting colors by preset object."""
        colors = get_preset_colors(PRESENTATION_PRESET)
        assert colors == PRESENTATION_PRESET.color_palette

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_limit_n_colors(self):
        """Test limiting number of colors returned."""
        colors = get_preset_colors("screen", n_colors=3)
        assert len(colors) == 3
        assert colors == SCREEN_PRESET.color_palette[:3]

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_cycle_colors_if_more_needed(self):
        """Test color cycling when requesting more than available."""
        palette = get_preset_colors("ieee_publication")
        assert len(palette) == 6

        # Request more colors than available
        colors = get_preset_colors("ieee_publication", n_colors=10)
        assert len(colors) == 10
        # Should cycle
        assert colors[0] == colors[6]  # First color repeats
        assert colors[1] == colors[7]  # Second color repeats

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_all_colors_when_n_none(self):
        """Test returning all colors when n_colors is None."""
        colors = get_preset_colors("presentation", n_colors=None)
        assert colors == PRESENTATION_PRESET.color_palette

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_invalid_preset_name(self):
        """Test ValueError for unknown preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_colors("nonexistent")

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_returns_reference_when_all_colors(self):
        """Test that function returns reference when n_colors=None."""
        colors = get_preset_colors("screen")
        # Without n_colors, returns direct reference to palette
        assert colors is SCREEN_PRESET.color_palette

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_returns_copy_when_sliced(self):
        """Test that function returns copy when n_colors specified."""
        colors1 = get_preset_colors("screen", n_colors=3)
        colors2 = get_preset_colors("screen", n_colors=3)

        # Should be different list objects
        assert colors1 is not colors2
        colors1[0] = "#FFFFFF"
        assert colors2[0] == SCREEN_PRESET.color_palette[0]  # Unchanged

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_zero_colors(self):
        """Test requesting zero colors."""
        colors = get_preset_colors("screen", n_colors=0)
        assert colors == []

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_one_color(self):
        """Test requesting single color."""
        colors = get_preset_colors("screen", n_colors=1)
        assert len(colors) == 1
        assert colors[0] == SCREEN_PRESET.color_palette[0]


class TestListPresets:
    """Tests for list_presets function."""

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_returns_list(self):
        """Test returns list of strings."""
        presets = list_presets()
        assert isinstance(presets, list)
        assert all(isinstance(p, str) for p in presets)

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_includes_all_presets(self):
        """Test includes all defined presets."""
        presets = list_presets()
        assert "ieee_publication" in presets
        assert "ieee_double_column" in presets
        assert "presentation" in presets
        assert "screen" in presets
        assert "print" in presets
        assert "dark" in presets

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_count(self):
        """Test returns correct number of presets."""
        presets = list_presets()
        assert len(presets) == 6

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_matches_registry_keys(self):
        """Test result matches PRESETS keys."""
        presets = list_presets()
        assert set(presets) == set(PRESETS.keys())


class TestCreateCustomPreset:
    """Tests for create_custom_preset function (VIS-024)."""

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_create_from_base_preset(self):
        """Test creating custom preset from base."""
        custom = create_custom_preset("my_preset", base_preset="screen")
        assert custom.name == "my_preset"
        assert isinstance(custom, VisualizationPreset)

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_inherits_from_base(self):
        """Test custom preset inherits from base."""
        custom = create_custom_preset("custom", base_preset="screen")
        # Should inherit screen settings
        assert custom.dpi == SCREEN_PRESET.dpi
        assert custom.font_family == SCREEN_PRESET.font_family
        assert custom.color_palette == SCREEN_PRESET.color_palette

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_override_dpi(self):
        """Test overriding DPI."""
        custom = create_custom_preset("custom", base_preset="screen", dpi=300)
        assert custom.dpi == 300
        assert custom.font_family == SCREEN_PRESET.font_family

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_override_figure_size(self):
        """Test overriding figure size."""
        custom = create_custom_preset("custom", base_preset="screen", figure_size=(8, 5))
        assert custom.figure_size == (8, 5)

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_override_multiple_attributes(self):
        """Test overriding multiple attributes."""
        custom = create_custom_preset(
            "custom",
            base_preset="ieee_publication",
            dpi=300,
            figure_size=(5, 3),
            font_family="monospace",
        )
        assert custom.dpi == 300
        assert custom.figure_size == (5, 3)
        assert custom.font_family == "monospace"

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_custom_description(self):
        """Test custom description."""
        custom = create_custom_preset(
            "custom", base_preset="screen", description="My custom preset"
        )
        assert custom.description == "My custom preset"

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_default_description(self):
        """Test default description generation."""
        custom = create_custom_preset("custom", base_preset="print")
        assert "print" in custom.description
        assert "Custom preset" in custom.description

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_custom_color_palette(self):
        """Test overriding color palette."""
        new_colors = ["#FF0000", "#00FF00", "#0000FF"]
        custom = create_custom_preset("custom", base_preset="screen", color_palette=new_colors)
        assert custom.color_palette == new_colors

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_custom_style_params(self):
        """Test overriding style params."""
        new_params = {"font.size": 15, "lines.linewidth": 2.5}
        custom = create_custom_preset("custom", base_preset="screen", style_params=new_params)
        assert custom.style_params == new_params

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_does_not_modify_base(self):
        """Test creating custom preset doesn't modify base."""
        original_dpi = SCREEN_PRESET.dpi
        custom = create_custom_preset("custom", base_preset="screen", dpi=999)
        assert SCREEN_PRESET.dpi == original_dpi
        assert custom.dpi == 999

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_invalid_base_preset(self):
        """Test ValueError for unknown base preset."""
        with pytest.raises(ValueError, match="Unknown base_preset"):
            create_custom_preset("custom", base_preset="nonexistent")

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_can_use_custom_preset_with_apply(self):
        """Test custom preset can be used with apply_preset."""
        custom = create_custom_preset("custom", base_preset="screen", dpi=150)

        with apply_preset(custom) as preset:
            assert preset.dpi == 150
            assert preset.name == "custom"

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_colorblind_safe_override(self):
        """Test overriding colorblind_safe flag."""
        custom = create_custom_preset("custom", base_preset="screen", colorblind_safe=False)
        assert custom.colorblind_safe is False

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_print_optimized_override(self):
        """Test overriding print_optimized flag."""
        custom = create_custom_preset("custom", base_preset="screen", print_optimized=True)
        assert custom.print_optimized is True


class TestPresetConsistency:
    """Tests for consistency across all presets."""

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_all_presets_have_valid_hex_colors(self):
        """Test all color palettes contain valid hex codes."""
        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")

        for name, preset in PRESETS.items():
            for color in preset.color_palette:
                assert hex_pattern.match(color), f"Invalid color in {name}: {color}"

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_all_presets_have_nonempty_palettes(self):
        """Test all presets have at least one color."""
        for name, preset in PRESETS.items():
            assert len(preset.color_palette) > 0, f"Empty palette in {name}"

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_all_presets_have_positive_dpi(self):
        """Test all presets have valid DPI values."""
        for name, preset in PRESETS.items():
            assert preset.dpi > 0, f"Invalid DPI in {name}"

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_all_presets_have_valid_figure_size(self):
        """Test all presets have valid figure sizes."""
        for name, preset in PRESETS.items():
            assert len(preset.figure_size) == 2, f"Invalid figure_size in {name}"
            assert preset.figure_size[0] > 0, f"Invalid width in {name}"
            assert preset.figure_size[1] > 0, f"Invalid height in {name}"

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_all_presets_have_style_params(self):
        """Test all presets have style parameters."""
        for name, preset in PRESETS.items():
            assert isinstance(preset.style_params, dict), f"Invalid style_params in {name}"
            assert len(preset.style_params) > 0, f"Empty style_params in {name}"

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_print_presets_disable_antialiasing(self):
        """Test print-optimized presets disable antialiasing."""
        for name, preset in PRESETS.items():
            if preset.print_optimized:
                assert preset.style_params.get("lines.antialiased") is False, (
                    f"Print preset {name} should disable antialiasing"
                )

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_all_presets_have_unique_names(self):
        """Test all preset names are unique."""
        names = [p.name for p in PRESETS.values()]
        assert len(names) == len(set(names))


class TestVisualizationPresetsIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_full_workflow_ieee_publication(self):
        """Test complete IEEE publication workflow."""
        # Get preset
        preset_name = "ieee_publication"

        # Apply preset and create figure
        with apply_preset(preset_name) as preset:
            fig, _ax = plt.subplots(figsize=preset.figure_size)

            # Use preset colors
            colors = preset.color_palette
            assert len(colors) == 6

            # Verify settings applied
            assert plt.rcParams["font.size"] == 8
            assert plt.rcParams["savefig.dpi"] == 600

            plt.close(fig)

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_custom_preset_workflow(self):
        """Test creating and using custom preset."""
        # Create custom preset
        custom = create_custom_preset(
            "my_preset",
            base_preset="screen",
            dpi=150,
            figure_size=(8, 6),
        )

        # Get colors
        colors = get_preset_colors(custom, n_colors=4)
        assert len(colors) == 4

        # Apply and use
        with apply_preset(custom) as preset:
            assert preset.dpi == 150
            assert plt.rcParams["figure.dpi"] == 96  # From base style_params

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_list_and_apply_all_presets(self):
        """Test listing and applying each preset."""
        preset_names = list_presets()

        for name in preset_names:
            with apply_preset(name) as preset:
                assert preset.name == name
                colors = get_preset_colors(name)
                assert len(colors) > 0


class TestVisualizationPresetsEdgeCases:
    """Additional edge case tests for comprehensive coverage."""

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_get_preset_colors_exact_match(self):
        """Test requesting exactly the number of colors available."""
        palette_size = len(SCREEN_PRESET.color_palette)
        colors = get_preset_colors("screen", n_colors=palette_size)
        assert len(colors) == palette_size
        assert colors == SCREEN_PRESET.color_palette[:palette_size]

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_apply_preset_empty_overrides(self):
        """Test applying preset with empty overrides dictionary."""
        with apply_preset("screen", overrides={}) as preset:
            assert preset is SCREEN_PRESET
            # Should still apply base settings
            assert plt.rcParams["font.size"] == 10

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_apply_preset_none_overrides(self):
        """Test applying preset with None overrides."""
        with apply_preset("screen", overrides=None) as preset:
            assert preset is SCREEN_PRESET

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_create_custom_preset_inherits_style_params_copy(self):
        """Test that custom preset gets copy of style params, not reference."""
        custom = create_custom_preset("custom", base_preset="screen")

        # Modify the custom preset's style params
        custom.style_params["font.size"] = 999

        # Original should be unchanged
        assert SCREEN_PRESET.style_params["font.size"] == 10

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_create_custom_preset_inherits_color_palette_copy(self):
        """Test that custom preset gets copy of color palette, not reference."""
        custom = create_custom_preset("custom", base_preset="screen")

        # Modify the custom preset's color palette
        original_first_color = SCREEN_PRESET.color_palette[0]
        custom.color_palette[0] = "#FFFFFF"

        # Original should be unchanged
        assert SCREEN_PRESET.color_palette[0] == original_first_color

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_preset_attribute_types(self):
        """Test that all preset attributes have correct types."""
        preset = SCREEN_PRESET
        assert isinstance(preset.name, str)
        assert isinstance(preset.description, str)
        assert isinstance(preset.style_params, dict)
        assert isinstance(preset.color_palette, list)
        assert isinstance(preset.dpi, int)
        assert isinstance(preset.figure_size, tuple)
        assert isinstance(preset.font_family, str)
        assert isinstance(preset.colorblind_safe, bool)
        assert isinstance(preset.print_optimized, bool)

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_apply_preset_multiple_overrides(self):
        """Test applying preset with multiple overrides."""
        overrides = {
            "font.size": 15,
            "lines.linewidth": 2.5,
            "axes.titlesize": 18,
        }

        with apply_preset("screen", overrides=overrides):
            assert plt.rcParams["font.size"] == 15
            assert plt.rcParams["lines.linewidth"] == 2.5
            assert plt.rcParams["axes.titlesize"] == 18

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_get_preset_colors_large_n_colors(self):
        """Test requesting many more colors than available (cycling)."""
        palette_size = len(IEEE_PUBLICATION_PRESET.color_palette)
        n_colors = palette_size * 5 + 2

        colors = get_preset_colors("ieee_publication", n_colors=n_colors)
        assert len(colors) == n_colors

        # Verify cycling pattern
        for i in range(n_colors):
            expected_color = IEEE_PUBLICATION_PRESET.color_palette[i % palette_size]
            assert colors[i] == expected_color

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_all_presets_have_description(self):
        """Test that all presets have non-empty descriptions."""
        for name, preset in PRESETS.items():
            assert preset.description, f"Preset {name} has empty description"
            assert len(preset.description) > 0

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_ieee_double_column_uses_copy_of_style_params(self):
        """Test that IEEE double-column has independent style params."""
        # IEEE_DOUBLE_COLUMN_PRESET uses .copy() of IEEE_PUBLICATION style params
        # Verify they are equal but not the same object
        assert (
            IEEE_DOUBLE_COLUMN_PRESET.style_params["font.size"]
            == IEEE_PUBLICATION_PRESET.style_params["font.size"]
        )
        # They should be different dict objects
        assert IEEE_DOUBLE_COLUMN_PRESET.style_params is not IEEE_PUBLICATION_PRESET.style_params

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_create_custom_preset_all_parameters(self):
        """Test creating custom preset with all parameters specified."""
        custom = create_custom_preset(
            name="fully_custom",
            base_preset="screen",
            description="Fully customized preset",
            style_params={"font.size": 20},
            color_palette=["#FF0000", "#00FF00"],
            dpi=200,
            figure_size=(9, 7),
            font_family="monospace",
            colorblind_safe=False,
            print_optimized=True,
        )

        assert custom.name == "fully_custom"
        assert custom.description == "Fully customized preset"
        assert custom.style_params == {"font.size": 20}
        assert custom.color_palette == ["#FF0000", "#00FF00"]
        assert custom.dpi == 200
        assert custom.figure_size == (9, 7)
        assert custom.font_family == "monospace"
        assert custom.colorblind_safe is False
        assert custom.print_optimized is True

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_dataclass_equality(self):
        """Test VisualizationPreset equality comparison."""
        preset1 = VisualizationPreset(
            name="test",
            description="Test",
            style_params={"font.size": 10},
            color_palette=["#000000"],
        )
        preset2 = VisualizationPreset(
            name="test",
            description="Test",
            style_params={"font.size": 10},
            color_palette=["#000000"],
        )
        # Dataclasses with same values should be equal
        assert preset1 == preset2

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_dataclass_inequality(self):
        """Test VisualizationPreset inequality comparison."""
        preset1 = VisualizationPreset(
            name="test1",
            description="Test",
            style_params={},
            color_palette=[],
        )
        preset2 = VisualizationPreset(
            name="test2",
            description="Test",
            style_params={},
            color_palette=[],
        )
        assert preset1 != preset2

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_apply_preset_exception_safety(self):
        """Test that context manager restores settings even on exception."""
        original_size = plt.rcParams["font.size"]

        try:
            with apply_preset("presentation"):
                assert plt.rcParams["font.size"] == 18
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still restore original settings
        assert plt.rcParams["font.size"] == original_size

    @pytest.mark.unit
    @pytest.mark.visualization
    def test_preset_colors_are_strings(self):
        """Test that all colors in all palettes are strings."""
        for name, preset in PRESETS.items():
            for color in preset.color_palette:
                assert isinstance(color, str), f"Non-string color in {name}: {type(color)}"
