"""Unit tests for colorblind-safe palettes.

Tests:
"""

import re

import matplotlib
import matplotlib.pyplot as plt
import pytest

# Set matplotlib backend before importing palettes
matplotlib.use("Agg")

from oscura.visualization.palettes import (
    LINE_STYLES,
    MARKER_STYLES,
    PALETTES,
    create_custom_palette,
    get_colormap,
    get_line_styles,
    get_palette,
    get_pass_fail_colors,
    get_pass_fail_symbols,
    show_palette,
    simulate_colorblindness,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.visualization,
    pytest.mark.usefixtures("cleanup_matplotlib"),
]


class TestGetPalette:
    """Tests for get_palette function."""

    @pytest.mark.unit
    def test_default_palette(self):
        """Test default colorblind_safe palette."""
        colors = get_palette()
        assert isinstance(colors, list)
        assert len(colors) == 8
        assert colors[0] == "#0173B2"  # Blue

    @pytest.mark.unit
    def test_colorblind_safe_palette(self):
        """Test explicit colorblind_safe palette."""
        colors = get_palette("colorblind_safe")
        assert len(colors) == 8
        assert all(isinstance(c, str) for c in colors)
        assert all(c.startswith("#") for c in colors)

    @pytest.mark.unit
    def test_colorblind8_palette(self):
        """Test Paul Tol's bright scheme."""
        colors = get_palette("colorblind8")
        assert len(colors) == 8
        assert colors[0] == "#4477AA"  # Blue

    @pytest.mark.unit
    def test_high_contrast_palette(self):
        """Test high contrast palette."""
        colors = get_palette("high_contrast")
        assert len(colors) == 8
        assert colors[0] == "#000000"  # Black

    @pytest.mark.unit
    def test_grayscale_palette(self):
        """Test grayscale palette."""
        colors = get_palette("grayscale")
        assert len(colors) == 5
        assert colors[0] == "#000000"  # Black
        assert colors[-1] == "#E0E0E0"  # Very Light Gray

    @pytest.mark.unit
    def test_returns_copy(self):
        """Test that palette returns a copy, not reference."""
        colors1 = get_palette("colorblind_safe")
        colors2 = get_palette("colorblind_safe")

        # Modify first list
        colors1[0] = "#FFFFFF"

        # Second list should be unchanged
        assert colors2[0] == "#0173B2"

    @pytest.mark.unit
    def test_invalid_palette_name(self):
        """Test that invalid palette name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown palette"):
            get_palette("nonexistent")  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_error_message_includes_valid_options(self):
        """Test that error message lists valid options."""
        with pytest.raises(ValueError, match="colorblind_safe"):
            get_palette("invalid")  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_all_hex_codes_valid(self):
        """Test that all colors in all palettes are valid hex codes."""
        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")

        for palette_name in ["colorblind_safe", "colorblind8", "high_contrast", "grayscale"]:
            colors = get_palette(palette_name)
            for color in colors:
                assert hex_pattern.match(color), f"Invalid hex code: {color}"


class TestGetColormap:
    """Tests for get_colormap function."""

    @pytest.mark.unit
    def test_default_colormap(self):
        """Test default viridis colormap."""
        pytest.importorskip("matplotlib")

        cmap = get_colormap()
        assert cmap is not None
        assert cmap.name == "viridis"

    @pytest.mark.unit
    def test_viridis_colormap(self):
        """Test explicit viridis colormap."""
        pytest.importorskip("matplotlib")

        cmap = get_colormap("viridis")
        assert cmap.name == "viridis"

    @pytest.mark.unit
    def test_cividis_colormap(self):
        """Test cividis colormap (CVD optimized)."""
        pytest.importorskip("matplotlib")

        cmap = get_colormap("cividis")
        assert cmap.name == "cividis"

    @pytest.mark.unit
    def test_plasma_colormap(self):
        """Test plasma colormap."""
        pytest.importorskip("matplotlib")

        cmap = get_colormap("plasma")
        assert cmap.name == "plasma"

    @pytest.mark.unit
    def test_inferno_colormap(self):
        """Test inferno colormap."""
        pytest.importorskip("matplotlib")

        cmap = get_colormap("inferno")
        assert cmap.name == "inferno"

    @pytest.mark.unit
    def test_magma_colormap(self):
        """Test magma colormap."""
        pytest.importorskip("matplotlib")

        cmap = get_colormap("magma")
        assert cmap.name == "magma"

    @pytest.mark.unit
    def test_invalid_colormap_name(self):
        """Test that invalid colormap name raises ValueError."""
        pytest.importorskip("matplotlib")

        with pytest.raises(ValueError, match="Unknown colormap"):
            get_colormap("rainbow")  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_colormap_callable(self):
        """Test that colormap can be called to get colors."""
        pytest.importorskip("matplotlib")

        cmap = get_colormap("viridis")

        # Get color at position 0.5
        color = cmap(0.5)
        assert len(color) == 4  # RGBA
        assert all(0 <= c <= 1 for c in color)


class TestGetLineStyles:
    """Tests for get_line_styles function."""

    @pytest.mark.unit
    def test_basic_line_styles(self):
        """Test basic line styles generation."""
        styles = get_line_styles(4)

        assert len(styles) == 4
        assert all(isinstance(s, dict) for s in styles)
        assert all("color" in s and "linestyle" in s for s in styles)

    @pytest.mark.unit
    def test_default_palette(self):
        """Test that default palette is used."""
        styles = get_line_styles(2)

        # Should use colorblind_safe palette by default
        assert styles[0]["color"] == "#0173B2"
        assert styles[1]["color"] == "#DE8F05"

    @pytest.mark.unit
    def test_custom_palette(self):
        """Test with custom palette."""
        styles = get_line_styles(2, palette="high_contrast")

        assert styles[0]["color"] == "#000000"
        assert styles[1]["color"] == "#E69F00"

    @pytest.mark.unit
    def test_cycle_colors(self):
        """Test that colors cycle when n_lines > palette size."""
        styles = get_line_styles(10, palette="colorblind_safe")

        # Should cycle back to first color
        assert styles[0]["color"] == styles[8]["color"]
        assert styles[1]["color"] == styles[9]["color"]

    @pytest.mark.unit
    def test_cycle_styles_true(self):
        """Test line style cycling when cycle_styles=True."""
        styles = get_line_styles(8, cycle_styles=True)

        # First 4 should cycle through LINE_STYLES
        expected_styles = ["solid", "dashed", "dotted", "dashdot"]
        for i in range(4):
            assert styles[i]["linestyle"] == expected_styles[i]

        # Should cycle back
        assert styles[4]["linestyle"] == "solid"

    @pytest.mark.unit
    def test_cycle_styles_false(self):
        """Test all solid when cycle_styles=False."""
        styles = get_line_styles(8, cycle_styles=False)

        # All should be solid
        assert all(s["linestyle"] == "solid" for s in styles)

    @pytest.mark.unit
    def test_zero_lines(self):
        """Test with zero lines."""
        styles = get_line_styles(0)
        assert styles == []

    @pytest.mark.unit
    def test_one_line(self):
        """Test with single line."""
        styles = get_line_styles(1)

        assert len(styles) == 1
        assert styles[0]["color"] == "#0173B2"
        assert styles[0]["linestyle"] == "solid"

    @pytest.mark.unit
    def test_many_lines(self):
        """Test with many lines (more than colors and styles)."""
        styles = get_line_styles(20)

        assert len(styles) == 20
        # Check cycling works correctly
        assert styles[0]["color"] == styles[8]["color"]
        assert styles[0]["linestyle"] == styles[4]["linestyle"]


class TestGetPassFailSymbols:
    """Tests for get_pass_fail_symbols function."""

    @pytest.mark.unit
    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        symbols = get_pass_fail_symbols()
        assert isinstance(symbols, dict)

    @pytest.mark.unit
    def test_has_pass_and_fail_keys(self):
        """Test that dict has pass and fail keys."""
        symbols = get_pass_fail_symbols()
        assert "pass" in symbols
        assert "fail" in symbols

    @pytest.mark.unit
    def test_pass_symbol(self):
        """Test pass symbol is checkmark."""
        symbols = get_pass_fail_symbols()
        assert symbols["pass"] == "✓"

    @pytest.mark.unit
    def test_fail_symbol(self):
        """Test fail symbol is cross."""
        symbols = get_pass_fail_symbols()
        assert symbols["fail"] == "✗"


class TestGetPassFailColors:
    """Tests for get_pass_fail_colors function."""

    @pytest.mark.unit
    def test_colorblind_safe_true(self):
        """Test colorblind-safe blue/orange colors."""
        colors = get_pass_fail_colors(colorblind_safe=True)

        assert colors["pass"] == "#0173B2"  # Blue
        assert colors["fail"] == "#DE8F05"  # Orange

    @pytest.mark.unit
    def test_colorblind_safe_false(self):
        """Test traditional green/red colors."""
        colors = get_pass_fail_colors(colorblind_safe=False)

        assert colors["pass"] == "#2CA02C"  # Green
        assert colors["fail"] == "#D62728"  # Red

    @pytest.mark.unit
    def test_default_is_colorblind_safe(self):
        """Test that default is colorblind-safe."""
        colors = get_pass_fail_colors()

        # Should default to colorblind_safe=True
        assert colors["pass"] == "#0173B2"
        assert colors["fail"] == "#DE8F05"

    @pytest.mark.unit
    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        colors = get_pass_fail_colors()
        assert isinstance(colors, dict)

    @pytest.mark.unit
    def test_has_pass_and_fail_keys(self):
        """Test that dict has pass and fail keys."""
        colors = get_pass_fail_colors()
        assert "pass" in colors
        assert "fail" in colors

    @pytest.mark.unit
    def test_all_valid_hex_codes(self):
        """Test that all colors are valid hex codes."""
        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")

        for colorblind_safe in [True, False]:
            colors = get_pass_fail_colors(colorblind_safe=colorblind_safe)
            for color in colors.values():
                assert hex_pattern.match(color)


class TestShowPalette:
    """Tests for show_palette function."""

    @pytest.mark.unit
    def test_show_discrete_palette(self, tmp_path):
        """Test showing discrete palette."""
        save_path = tmp_path / "palette.png"
        show_palette("colorblind_safe", save_path=str(save_path))

        assert save_path.exists()
        plt.close("all")

    @pytest.mark.unit
    def test_show_colormap(self, tmp_path):
        """Test showing continuous colormap."""
        save_path = tmp_path / "colormap.png"
        show_palette("viridis", save_path=str(save_path))

        assert save_path.exists()
        plt.close("all")

    @pytest.mark.unit
    def test_all_palettes(self, tmp_path):
        """Test showing all discrete palettes."""
        for palette_name in ["colorblind_safe", "colorblind8", "high_contrast", "grayscale"]:
            save_path = tmp_path / f"{palette_name}.png"
            show_palette(palette_name, save_path=str(save_path))
            assert save_path.exists()

        plt.close("all")

    @pytest.mark.unit
    def test_all_colormaps(self, tmp_path):
        """Test showing all colormaps."""
        for cmap_name in ["viridis", "cividis", "plasma", "inferno", "magma"]:
            save_path = tmp_path / f"{cmap_name}.png"
            show_palette(cmap_name, save_path=str(save_path))
            assert save_path.exists()

        plt.close("all")

    @pytest.mark.unit
    def test_invalid_name(self):
        """Test that invalid name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown palette or colormap"):
            show_palette("invalid_name")

        plt.close("all")

    @pytest.mark.unit
    def test_without_save_path(self, monkeypatch):
        """Test showing palette without saving (mocked)."""
        # Mock plt.show to avoid displaying
        show_called = []
        monkeypatch.setattr(plt, "show", lambda: show_called.append(True))

        show_palette("colorblind_safe", save_path=None)

        assert len(show_called) == 1
        plt.close("all")


class TestCreateCustomPalette:
    """Tests for create_custom_palette function."""

    @pytest.mark.unit
    def test_basic_custom_palette(self):
        """Test creating basic custom palette."""
        colors = ["#FF0000", "#00FF00", "#0000FF"]
        custom = create_custom_palette(colors, name="rgb")

        assert len(custom) == 3
        assert custom[0] == "#FF0000"
        assert custom[1] == "#00FF00"
        assert custom[2] == "#0000FF"

    @pytest.mark.unit
    def test_normalizes_to_uppercase(self):
        """Test that hex codes are normalized to uppercase."""
        colors = ["#ff0000", "#00ff00", "#0000ff"]
        custom = create_custom_palette(colors, name="rgb_lower")

        assert custom[0] == "#FF0000"
        assert custom[1] == "#00FF00"
        assert custom[2] == "#0000FF"

    @pytest.mark.unit
    def test_registers_palette(self):
        """Test that custom palette is registered in PALETTES."""
        colors = ["#AABBCC", "#DDEEFF"]
        name = "test_palette"

        create_custom_palette(colors, name=name)

        assert name in PALETTES
        assert PALETTES[name][0] == "#AABBCC"

    @pytest.mark.unit
    def test_invalid_hex_code_no_hash(self):
        """Test that hex code without # raises ValueError."""
        colors = ["FF0000"]  # Missing #

        with pytest.raises(ValueError, match="Invalid hex color code"):
            create_custom_palette(colors)

    @pytest.mark.unit
    def test_invalid_hex_code_wrong_length(self):
        """Test that wrong length hex code raises ValueError."""
        colors = ["#FF00"]  # Too short

        with pytest.raises(ValueError, match="Invalid hex color code"):
            create_custom_palette(colors)

    @pytest.mark.unit
    def test_invalid_hex_code_wrong_chars(self):
        """Test that invalid characters raise ValueError."""
        colors = ["#GGHHII"]  # Invalid hex chars

        with pytest.raises(ValueError, match="Invalid hex color code"):
            create_custom_palette(colors)

    @pytest.mark.unit
    def test_empty_list(self):
        """Test creating palette with empty list."""
        colors = []
        custom = create_custom_palette(colors)

        assert custom == []

    @pytest.mark.unit
    def test_single_color(self):
        """Test creating palette with single color."""
        colors = ["#123456"]
        custom = create_custom_palette(colors)

        assert len(custom) == 1
        assert custom[0] == "#123456"

    @pytest.mark.unit
    def test_mixed_case_hex_codes(self):
        """Test with mixed case hex codes."""
        colors = ["#AbCdEf", "#12345"]  # Mixed case, one invalid

        # First one should work
        custom = create_custom_palette(colors[:1])
        assert custom[0] == "#ABCDEF"


class TestSimulateColorblindness:
    """Tests for simulate_colorblindness function."""

    @pytest.mark.unit
    def test_protanopia_simulation(self):
        """Test protanopia (red-blind) simulation."""
        red = "#FF0000"
        simulated = simulate_colorblindness(red, deficiency="protanopia")

        # Should be a valid hex code
        assert re.match(r"^#[0-9A-F]{6}$", simulated)
        # Red should appear different (brownish)
        assert simulated != red

    @pytest.mark.unit
    def test_deuteranopia_simulation(self):
        """Test deuteranopia (green-blind) simulation."""
        green = "#00FF00"
        simulated = simulate_colorblindness(green, deficiency="deuteranopia")

        assert re.match(r"^#[0-9A-F]{6}$", simulated)
        assert simulated != green

    @pytest.mark.unit
    def test_tritanopia_simulation(self):
        """Test tritanopia (blue-blind) simulation."""
        blue = "#0000FF"
        simulated = simulate_colorblindness(blue, deficiency="tritanopia")

        assert re.match(r"^#[0-9A-F]{6}$", simulated)
        assert simulated != blue

    @pytest.mark.unit
    def test_default_deficiency(self):
        """Test default deficiency is deuteranopia."""
        color = "#FF00FF"
        simulated = simulate_colorblindness(color)

        # Should use deuteranopia by default
        assert re.match(r"^#[0-9A-F]{6}$", simulated)

    @pytest.mark.unit
    def test_invalid_deficiency(self):
        """Test that invalid deficiency raises ValueError."""
        with pytest.raises(ValueError, match="Unknown deficiency"):
            simulate_colorblindness("#FF0000", deficiency="invalid")  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_lowercase_hex_code(self):
        """Test with lowercase hex code."""
        color = "#ff00ff"
        simulated = simulate_colorblindness(color, deficiency="deuteranopia")

        assert re.match(r"^#[0-9A-F]{6}$", simulated)

    @pytest.mark.unit
    def test_black_unchanged_mostly(self):
        """Test that black stays mostly black."""
        black = "#000000"

        for deficiency in ["protanopia", "deuteranopia", "tritanopia"]:
            simulated = simulate_colorblindness(black, deficiency=deficiency)
            # Black should remain black or very close
            assert simulated == "#000000"

    @pytest.mark.unit
    def test_white_unchanged_mostly(self):
        """Test that white stays mostly white."""
        white = "#FFFFFF"

        for deficiency in ["protanopia", "deuteranopia", "tritanopia"]:
            simulated = simulate_colorblindness(white, deficiency=deficiency)
            # White should remain white or very close
            assert simulated == "#FFFFFF"

    @pytest.mark.unit
    def test_output_format_uppercase(self):
        """Test that output is uppercase hex."""
        color = "#abc123"
        simulated = simulate_colorblindness(color, deficiency="deuteranopia")

        # Should be uppercase
        assert simulated == simulated.upper()

    @pytest.mark.unit
    def test_red_green_confusion_deuteranopia(self):
        """Test that red and green appear similar under deuteranopia."""
        red = "#FF0000"
        green = "#00FF00"

        red_sim = simulate_colorblindness(red, deficiency="deuteranopia")
        green_sim = simulate_colorblindness(green, deficiency="deuteranopia")

        # They should be more similar than original colors
        # Extract RGB values
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

        red_rgb = hex_to_rgb(red)
        green_rgb = hex_to_rgb(green)
        red_sim_rgb = hex_to_rgb(red_sim)
        green_sim_rgb = hex_to_rgb(green_sim)

        # Original distance
        orig_dist = sum((a - b) ** 2 for a, b in zip(red_rgb, green_rgb, strict=False))
        # Simulated distance
        sim_dist = sum((a - b) ** 2 for a, b in zip(red_sim_rgb, green_sim_rgb, strict=False))

        # Simulated should be closer
        assert sim_dist < orig_dist

    @pytest.mark.unit
    def test_clipping_at_boundaries(self):
        """Test that extreme values are clipped properly."""
        # Test with various colors to ensure no overflow
        colors = ["#FFFFFF", "#000000", "#FF0000", "#00FF00", "#0000FF"]

        for color in colors:
            for deficiency in ["protanopia", "deuteranopia", "tritanopia"]:
                simulated = simulate_colorblindness(color, deficiency=deficiency)

                # Extract RGB and verify in valid range
                hex_color = simulated.lstrip("#")
                r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

                assert 0 <= r <= 255
                assert 0 <= g <= 255
                assert 0 <= b <= 255


class TestModuleConstants:
    """Tests for module-level constants."""

    @pytest.mark.unit
    def test_line_styles_constant(self):
        """Test LINE_STYLES constant."""
        assert LINE_STYLES == ["solid", "dashed", "dotted", "dashdot"]
        assert len(LINE_STYLES) == 4

    @pytest.mark.unit
    def test_marker_styles_constant(self):
        """Test MARKER_STYLES constant."""
        assert MARKER_STYLES == ["o", "s", "^", "D", "v", "p", "*", "h"]
        assert len(MARKER_STYLES) == 8

    @pytest.mark.unit
    def test_palettes_constant_structure(self):
        """Test PALETTES constant structure."""
        assert isinstance(PALETTES, dict)
        assert "colorblind_safe" in PALETTES
        assert "colorblind8" in PALETTES
        assert "high_contrast" in PALETTES
        assert "grayscale" in PALETTES

    @pytest.mark.unit
    def test_palettes_are_lists(self):
        """Test that all palette values are lists."""
        for palette in PALETTES.values():
            assert isinstance(palette, list)

    @pytest.mark.unit
    def test_palettes_contain_hex_codes(self):
        """Test that all palettes contain valid hex codes."""
        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")

        for palette_name, colors in PALETTES.items():
            for color in colors:
                assert hex_pattern.match(color), f"Invalid hex in {palette_name}: {color}"


class TestVisualizationPalettesIntegration:
    """Integration tests combining multiple functions."""

    @pytest.mark.unit
    def test_palette_to_line_styles(self):
        """Test using palette with line styles."""
        # Get palette and create line styles from it
        palette_name = "colorblind8"
        colors = get_palette(palette_name)
        styles = get_line_styles(len(colors), palette=palette_name)

        assert len(styles) == len(colors)
        for _i, (color, style) in enumerate(zip(colors, styles, strict=False)):
            assert style["color"] == color

    @pytest.mark.unit
    def test_custom_palette_usage(self):
        """Test creating and using custom palette."""
        colors = ["#112233", "#445566", "#778899"]
        name = "custom_test"

        create_custom_palette(colors, name=name)
        retrieved = get_palette(name)  # type: ignore[arg-type]

        assert retrieved[0] == "#112233"

    @pytest.mark.unit
    def test_colorblind_simulation_on_palette(self):
        """Test simulating colorblindness on entire palette."""
        colors = get_palette("colorblind_safe")

        simulated = [simulate_colorblindness(color, deficiency="deuteranopia") for color in colors]

        assert len(simulated) == len(colors)
        # All should be valid hex codes
        hex_pattern = re.compile(r"^#[0-9A-F]{6}$")
        assert all(hex_pattern.match(c) for c in simulated)

    @pytest.mark.unit
    def test_pass_fail_symbols_and_colors_together(self):
        """Test using pass/fail symbols and colors together."""
        symbols = get_pass_fail_symbols()
        colors = get_pass_fail_colors(colorblind_safe=True)

        # Should be able to combine them
        pass_msg = f"{symbols['pass']} PASS (color: {colors['pass']})"
        fail_msg = f"{symbols['fail']} FAIL (color: {colors['fail']})"

        assert "✓" in pass_msg
        assert "✗" in fail_msg
        assert "#0173B2" in pass_msg
        assert "#DE8F05" in fail_msg
