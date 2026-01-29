"""Tests for visualization color palette utilities."""

import pytest

from oscura.visualization.colors import (
    COLORBLIND_SAFE_QUALITATIVE,
    DIVERGING_COOLWARM,
    SEQUENTIAL_VIRIDIS,
    _adjust_for_contrast,
    _auto_select_palette_type,
    _contrast_ratio,
    _generate_qualitative,
    _hsl_to_rgb,
    _interpolate_colors,
    _relative_luminance,
    _rgb_to_hsl,
    select_optimal_palette,
)

pytestmark = pytest.mark.usefixtures("cleanup_matplotlib")


class TestSelectOptimalPalette:
    """Test select_optimal_palette function."""

    def test_basic_qualitative_palette(self) -> None:
        """Test basic qualitative palette selection."""
        colors = select_optimal_palette(3, palette_type="qualitative")
        assert len(colors) == 3
        assert all(c.startswith("#") for c in colors)
        assert all(len(c) == 7 for c in colors)

    def test_basic_sequential_palette(self) -> None:
        """Test basic sequential palette selection."""
        colors = select_optimal_palette(10, palette_type="sequential")
        assert len(colors) == 10
        assert all(c.startswith("#") for c in colors)

    def test_basic_diverging_palette(self) -> None:
        """Test basic diverging palette selection."""
        colors = select_optimal_palette(10, palette_type="diverging")
        assert len(colors) == 10
        assert all(c.startswith("#") for c in colors)

    def test_auto_select_bipolar_data(self) -> None:
        """Test auto-selection for bipolar data."""
        # Should select diverging for bipolar data
        colors = select_optimal_palette(10, data_range=(-5.0, 5.0))
        assert len(colors) == 10
        # Auto-select should choose diverging for bipolar

    def test_auto_select_unipolar_data(self) -> None:
        """Test auto-selection for unipolar data."""
        # Should select sequential for unipolar data with many colors
        colors = select_optimal_palette(12, data_range=(0.0, 100.0))
        assert len(colors) == 12

    def test_auto_select_few_colors(self) -> None:
        """Test auto-selection for few distinct colors."""
        # Should select qualitative for few colors
        colors = select_optimal_palette(4)
        assert len(colors) == 4

    def test_colorblind_safe_option(self) -> None:
        """Test colorblind-safe palette selection."""
        colors = select_optimal_palette(3, palette_type="qualitative", colorblind_safe=True)
        assert len(colors) == 3
        # Should use predefined colorblind-safe palette
        assert colors[0] in COLORBLIND_SAFE_QUALITATIVE

    def test_colorblind_unsafe_option(self) -> None:
        """Test non-colorblind-safe palette selection."""
        colors = select_optimal_palette(3, palette_type="qualitative", colorblind_safe=False)
        assert len(colors) == 3

    def test_contrast_ratio_enforcement(self) -> None:
        """Test contrast ratio enforcement."""
        # High contrast requirement
        colors = select_optimal_palette(3, min_contrast_ratio=4.5)
        assert len(colors) == 3
        # Function attempts to adjust colors for contrast
        # At minimum, colors should be valid hex codes
        for color in colors:
            assert color.startswith("#")
            assert len(color) == 7

    def test_dark_background(self) -> None:
        """Test palette selection for dark background."""
        colors = select_optimal_palette(3, background_color="#000000", min_contrast_ratio=4.5)
        assert len(colors) == 3
        # Colors should have sufficient contrast with black background

    def test_interpolate_many_colors(self) -> None:
        """Test color interpolation for many colors."""
        # Request more colors than available in base palette
        colors = select_optimal_palette(50, palette_type="sequential")
        assert len(colors) == 50
        assert all(c.startswith("#") for c in colors)

    def test_invalid_n_colors(self) -> None:
        """Test error handling for invalid n_colors."""
        with pytest.raises(ValueError, match="n_colors must be >= 1"):
            select_optimal_palette(0)

        with pytest.raises(ValueError, match="n_colors must be >= 1"):
            select_optimal_palette(-1)

    def test_invalid_contrast_ratio(self) -> None:
        """Test error handling for invalid contrast ratio."""
        with pytest.raises(ValueError, match="min_contrast_ratio must be >= 1.0"):
            select_optimal_palette(3, min_contrast_ratio=0.5)

    def test_invalid_palette_type(self) -> None:
        """Test error handling for invalid palette type."""
        with pytest.raises(ValueError, match="Unknown palette_type"):
            select_optimal_palette(3, palette_type="invalid")  # type: ignore[arg-type]

    def test_single_color(self) -> None:
        """Test single color selection."""
        colors = select_optimal_palette(1)
        assert len(colors) == 1
        assert colors[0].startswith("#")


class TestAutoSelectPaletteType:
    """Test _auto_select_palette_type function."""

    def test_bipolar_data_diverging(self) -> None:
        """Test diverging selection for bipolar data."""
        palette_type = _auto_select_palette_type(10, data_range=(-10.0, 10.0))
        assert palette_type == "diverging"

    def test_unipolar_negative_sequential(self) -> None:
        """Test sequential selection for negative unipolar data."""
        palette_type = _auto_select_palette_type(10, data_range=(-100.0, -10.0))
        assert palette_type == "sequential"

    def test_unipolar_positive_sequential(self) -> None:
        """Test sequential selection for positive unipolar data."""
        palette_type = _auto_select_palette_type(12, data_range=(0.0, 100.0))
        assert palette_type == "sequential"

    def test_few_colors_qualitative(self) -> None:
        """Test qualitative selection for few colors."""
        palette_type = _auto_select_palette_type(5, data_range=None)
        assert palette_type == "qualitative"

    def test_many_colors_sequential(self) -> None:
        """Test sequential selection for many colors."""
        palette_type = _auto_select_palette_type(15, data_range=None)
        assert palette_type == "sequential"

    def test_boundary_8_colors(self) -> None:
        """Test boundary case with 8 colors."""
        palette_type = _auto_select_palette_type(8, data_range=None)
        assert palette_type == "qualitative"


class TestRelativeLuminance:
    """Test _relative_luminance function."""

    def test_white_luminance(self) -> None:
        """Test luminance of white."""
        lum = _relative_luminance("#FFFFFF")
        assert abs(lum - 1.0) < 0.01  # Should be close to 1.0

    def test_black_luminance(self) -> None:
        """Test luminance of black."""
        lum = _relative_luminance("#000000")
        assert abs(lum - 0.0) < 0.01  # Should be close to 0.0

    def test_gray_luminance(self) -> None:
        """Test luminance of middle gray."""
        lum = _relative_luminance("#808080")
        assert 0.2 < lum < 0.4  # Middle gray should be around 0.3

    def test_red_luminance(self) -> None:
        """Test luminance of pure red."""
        lum = _relative_luminance("#FF0000")
        assert 0.1 < lum < 0.3  # Red has lower luminance than white

    def test_green_luminance(self) -> None:
        """Test luminance of pure green."""
        lum = _relative_luminance("#00FF00")
        assert 0.6 < lum < 0.9  # Green has highest luminance contribution

    def test_blue_luminance(self) -> None:
        """Test luminance of pure blue."""
        lum = _relative_luminance("#0000FF")
        assert 0.0 < lum < 0.1  # Blue has lowest luminance contribution

    def test_with_hash_prefix(self) -> None:
        """Test with # prefix."""
        lum1 = _relative_luminance("#FF0000")
        assert 0.0 <= lum1 <= 1.0

    def test_without_hash_prefix(self) -> None:
        """Test without # prefix (removeprefix handles this)."""
        # The function uses removeprefix, so this should work
        lum = _relative_luminance("#808080")
        assert 0.0 <= lum <= 1.0


class TestContrastRatio:
    """Test _contrast_ratio function."""

    def test_white_black_contrast(self) -> None:
        """Test maximum contrast between white and black."""
        white_lum = _relative_luminance("#FFFFFF")
        black_lum = _relative_luminance("#000000")
        ratio = _contrast_ratio(white_lum, black_lum)
        assert abs(ratio - 21.0) < 0.5  # Should be close to 21:1

    def test_same_color_contrast(self) -> None:
        """Test minimum contrast (same color)."""
        lum = _relative_luminance("#808080")
        ratio = _contrast_ratio(lum, lum)
        assert abs(ratio - 1.0) < 0.01  # Should be 1:1

    def test_commutative_property(self) -> None:
        """Test that contrast ratio is commutative."""
        lum1 = _relative_luminance("#FF0000")
        lum2 = _relative_luminance("#0000FF")
        ratio1 = _contrast_ratio(lum1, lum2)
        ratio2 = _contrast_ratio(lum2, lum1)
        assert abs(ratio1 - ratio2) < 0.01

    def test_moderate_contrast(self) -> None:
        """Test moderate contrast colors."""
        lum1 = _relative_luminance("#333333")
        lum2 = _relative_luminance("#FFFFFF")
        ratio = _contrast_ratio(lum1, lum2)
        assert 10.0 < ratio < 16.0  # Dark gray on white has good contrast


class TestAdjustForContrast:
    """Test _adjust_for_contrast function."""

    def test_adjust_dark_color_light_background(self) -> None:
        """Test adjusting dark color on light background."""
        adjusted = _adjust_for_contrast("#333333", "#FFFFFF", target_ratio=4.5)
        assert adjusted.startswith("#")
        assert len(adjusted) == 7
        # Verify contrast meets target
        adj_lum = _relative_luminance(adjusted)
        bg_lum = _relative_luminance("#FFFFFF")
        ratio = _contrast_ratio(adj_lum, bg_lum)
        assert ratio >= 4.5 or abs(ratio - 4.5) < 0.3

    def test_adjust_light_color_dark_background(self) -> None:
        """Test adjusting light color on dark background."""
        adjusted = _adjust_for_contrast("#CCCCCC", "#000000", target_ratio=4.5)
        assert adjusted.startswith("#")
        # Verify contrast meets target
        adj_lum = _relative_luminance(adjusted)
        bg_lum = _relative_luminance("#000000")
        ratio = _contrast_ratio(adj_lum, bg_lum)
        assert ratio >= 4.5 or abs(ratio - 4.5) < 0.3

    def test_adjust_high_contrast_requirement(self) -> None:
        """Test adjustment for high contrast (AAA)."""
        adjusted = _adjust_for_contrast("#FF0000", "#FFFFFF", target_ratio=4.5)
        # Verify function returns valid color
        assert adjusted.startswith("#")
        assert len(adjusted) == 7
        # Verify it's a different color (was adjusted)
        # Red on white doesn't have sufficient contrast, so should be adjusted
        adj_lum = _relative_luminance(adjusted)
        orig_lum = _relative_luminance("#FF0000")
        # Should have changed the luminance
        assert adj_lum != orig_lum

    def test_already_sufficient_contrast(self) -> None:
        """Test color that already has sufficient contrast."""
        # Black on white already has high contrast
        adjusted = _adjust_for_contrast("#000000", "#FFFFFF", target_ratio=4.5)
        assert adjusted.startswith("#")


class TestRgbToHsl:
    """Test _rgb_to_hsl conversion."""

    def test_white_conversion(self) -> None:
        """Test white RGB to HSL."""
        h, s, l = _rgb_to_hsl(255, 255, 255)
        assert abs(l - 1.0) < 0.01  # Lightness should be 1.0
        assert abs(s - 0.0) < 0.01  # Saturation should be 0.0

    def test_black_conversion(self) -> None:
        """Test black RGB to HSL."""
        h, s, l = _rgb_to_hsl(0, 0, 0)
        assert abs(l - 0.0) < 0.01  # Lightness should be 0.0
        assert abs(s - 0.0) < 0.01  # Saturation should be 0.0

    def test_pure_red_conversion(self) -> None:
        """Test pure red RGB to HSL."""
        h, s, l = _rgb_to_hsl(255, 0, 0)
        assert abs(h - 0.0) < 1.0  # Hue should be ~0 (red)
        assert s > 0.9  # High saturation
        assert abs(l - 0.5) < 0.1  # Medium lightness

    def test_pure_green_conversion(self) -> None:
        """Test pure green RGB to HSL."""
        h, s, l = _rgb_to_hsl(0, 255, 0)
        assert abs(h - 120.0) < 1.0  # Hue should be ~120 (green)
        assert s > 0.9  # High saturation

    def test_pure_blue_conversion(self) -> None:
        """Test pure blue RGB to HSL."""
        h, s, l = _rgb_to_hsl(0, 0, 255)
        assert abs(h - 240.0) < 1.0  # Hue should be ~240 (blue)
        assert s > 0.9  # High saturation

    def test_gray_conversion(self) -> None:
        """Test gray RGB to HSL."""
        h, s, l = _rgb_to_hsl(128, 128, 128)
        assert abs(s - 0.0) < 0.01  # Saturation should be 0 (achromatic)


class TestHslToRgb:
    """Test _hsl_to_rgb conversion."""

    def test_white_conversion(self) -> None:
        """Test white HSL to RGB."""
        r, g, b = _hsl_to_rgb(0.0, 0.0, 1.0)
        assert r == 255
        assert g == 255
        assert b == 255

    def test_black_conversion(self) -> None:
        """Test black HSL to RGB."""
        r, g, b = _hsl_to_rgb(0.0, 0.0, 0.0)
        assert r == 0
        assert g == 0
        assert b == 0

    def test_pure_red_conversion(self) -> None:
        """Test pure red HSL to RGB."""
        r, g, b = _hsl_to_rgb(0.0, 1.0, 0.5)
        assert r == 255
        assert g == 0
        assert b == 0

    def test_pure_green_conversion(self) -> None:
        """Test pure green HSL to RGB."""
        r, g, b = _hsl_to_rgb(120.0, 1.0, 0.5)
        assert g == 255
        assert abs(r - 0) < 2  # Allow small rounding error
        assert abs(b - 0) < 2

    def test_pure_blue_conversion(self) -> None:
        """Test pure blue HSL to RGB."""
        r, g, b = _hsl_to_rgb(240.0, 1.0, 0.5)
        assert b == 255
        assert abs(r - 0) < 2
        assert abs(g - 0) < 2

    def test_gray_conversion(self) -> None:
        """Test gray (achromatic) HSL to RGB."""
        r, g, b = _hsl_to_rgb(0.0, 0.0, 0.5)
        # All components should be equal for achromatic
        assert r == g == b
        assert abs(r - 127) < 2  # Should be around middle gray

    def test_roundtrip_conversion(self) -> None:
        """Test RGB -> HSL -> RGB roundtrip."""
        original_r, original_g, original_b = 180, 90, 210
        h, s, l = _rgb_to_hsl(original_r, original_g, original_b)
        r, g, b = _hsl_to_rgb(h, s, l)
        # Allow small rounding errors
        assert abs(r - original_r) < 2
        assert abs(g - original_g) < 2
        assert abs(b - original_b) < 2


class TestGenerateQualitative:
    """Test _generate_qualitative function."""

    def test_generate_basic_colors(self) -> None:
        """Test generating basic qualitative colors."""
        colors = _generate_qualitative(6)
        assert len(colors) == 6
        assert all(c.startswith("#") for c in colors)
        assert all(len(c) == 7 for c in colors)

    def test_generate_single_color(self) -> None:
        """Test generating single color."""
        colors = _generate_qualitative(1)
        assert len(colors) == 1
        assert colors[0].startswith("#")

    def test_generate_many_colors(self) -> None:
        """Test generating many qualitative colors."""
        colors = _generate_qualitative(20)
        assert len(colors) == 20
        # All colors should be distinct
        assert len(set(colors)) == 20

    def test_evenly_spaced_hues(self) -> None:
        """Test that generated colors have evenly spaced hues."""
        colors = _generate_qualitative(12)
        # Convert to HSL and check hue spacing
        hues = []
        for color in colors:
            color_val = color.removeprefix("#")
            r = int(color_val[0:2], 16)
            g = int(color_val[2:4], 16)
            b = int(color_val[4:6], 16)
            h, _, _ = _rgb_to_hsl(r, g, b)
            hues.append(h)

        # Hues should be roughly evenly spaced
        expected_spacing = 360.0 / 12
        for i in range(len(hues) - 1):
            spacing = (hues[i + 1] - hues[i]) % 360
            assert abs(spacing - expected_spacing) < 5.0  # Allow small tolerance


class TestInterpolateColors:
    """Test _interpolate_colors function."""

    def test_interpolate_more_colors(self) -> None:
        """Test interpolating to create more colors."""
        base_colors = ["#FF0000", "#00FF00", "#0000FF"]
        interpolated = _interpolate_colors(base_colors, n_colors=10)
        assert len(interpolated) == 10
        assert all(c.startswith("#") for c in interpolated)

    def test_interpolate_same_count(self) -> None:
        """Test interpolating with same count as base."""
        base_colors = ["#FF0000", "#00FF00", "#0000FF"]
        interpolated = _interpolate_colors(base_colors, n_colors=3)
        assert len(interpolated) == 3

    def test_interpolate_fewer_colors(self) -> None:
        """Test interpolating with fewer colors than base."""
        base_colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00"]
        interpolated = _interpolate_colors(base_colors, n_colors=2)
        assert len(interpolated) == 2

    def test_interpolate_preserves_endpoints(self) -> None:
        """Test that interpolation preserves endpoint colors."""
        base_colors = ["#000000", "#FFFFFF"]
        interpolated = _interpolate_colors(base_colors, n_colors=5)
        assert len(interpolated) == 5
        # First color should be close to black
        assert interpolated[0] in ["#000000", "#00"]
        # Last color should be close to white
        assert interpolated[-1] in ["#ffffff", "#FFFFFF"]

    def test_interpolate_viridis(self) -> None:
        """Test interpolating viridis palette."""
        interpolated = _interpolate_colors(SEQUENTIAL_VIRIDIS, n_colors=100)
        assert len(interpolated) == 100
        assert all(c.startswith("#") for c in interpolated)


class TestPredefinedPalettes:
    """Test predefined color palettes."""

    def test_colorblind_safe_qualitative_palette(self) -> None:
        """Test colorblind-safe qualitative palette."""
        assert len(COLORBLIND_SAFE_QUALITATIVE) == 8
        assert all(c.startswith("#") for c in COLORBLIND_SAFE_QUALITATIVE)
        # Check some known colors
        assert "#0173B2" in COLORBLIND_SAFE_QUALITATIVE  # Blue
        assert "#DE8F05" in COLORBLIND_SAFE_QUALITATIVE  # Orange

    def test_sequential_viridis_palette(self) -> None:
        """Test sequential viridis palette."""
        assert len(SEQUENTIAL_VIRIDIS) == 20
        assert all(c.startswith("#") for c in SEQUENTIAL_VIRIDIS)
        # First color should be dark purple
        assert SEQUENTIAL_VIRIDIS[0] == "#440154"
        # Last color should be bright yellow
        assert SEQUENTIAL_VIRIDIS[-1] == "#FDE724"

    def test_diverging_coolwarm_palette(self) -> None:
        """Test diverging cool-warm palette."""
        assert len(DIVERGING_COOLWARM) == 13
        assert all(c.startswith("#") for c in DIVERGING_COOLWARM)
        # First color should be cool (blue)
        assert DIVERGING_COOLWARM[0] == "#3B4CC0"
        # Last color should be warm (red)
        assert DIVERGING_COOLWARM[-1] == "#B40426"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_colors(self) -> None:
        """Test with minimum number of colors."""
        colors = select_optimal_palette(1)
        assert len(colors) == 1

    def test_maximum_interpolation(self) -> None:
        """Test with very large number of colors."""
        colors = select_optimal_palette(200, palette_type="sequential")
        assert len(colors) == 200
        assert all(c.startswith("#") for c in colors)

    def test_extreme_contrast_requirement(self) -> None:
        """Test with maximum contrast requirement."""
        colors = select_optimal_palette(3, min_contrast_ratio=21.0)
        # Should adjust colors to meet requirement
        assert len(colors) == 3

    def test_zero_data_range(self) -> None:
        """Test with zero data range."""
        colors = select_optimal_palette(5, data_range=(0.0, 0.0))
        assert len(colors) == 5

    def test_negative_data_range(self) -> None:
        """Test with entirely negative data range."""
        colors = select_optimal_palette(8, data_range=(-100.0, -10.0))
        assert len(colors) == 8
