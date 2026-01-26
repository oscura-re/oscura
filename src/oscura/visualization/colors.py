"""Color palette selection and accessibility utilities.

This module provides intelligent color palette selection based on data
characteristics and accessibility requirements with WCAG contrast checking.


Example:
    >>> from oscura.visualization.colors import select_optimal_palette
    >>> colors = select_optimal_palette(n_channels=3, palette_type="qualitative")

References:
    WCAG 2.1 contrast guidelines
    Colorblind-safe palette design (Brettel 1997)
    ColorBrewer schemes
"""

from typing import Literal

import numpy as np

# Predefined colorblind-safe palettes
COLORBLIND_SAFE_QUALITATIVE = [
    "#0173B2",  # Blue
    "#DE8F05",  # Orange
    "#029E73",  # Green
    "#CC78BC",  # Purple
    "#CA9161",  # Brown
    "#949494",  # Gray
    "#ECE133",  # Yellow
    "#56B4E9",  # Light blue
]

SEQUENTIAL_VIRIDIS = [
    "#440154",
    "#481567",
    "#482677",
    "#453781",
    "#404788",
    "#39568C",
    "#33638D",
    "#2D708E",
    "#287D8E",
    "#238A8D",
    "#1F968B",
    "#20A387",
    "#29AF7F",
    "#3CBB75",
    "#55C667",
    "#73D055",
    "#95D840",
    "#B8DE29",
    "#DCE319",
    "#FDE724",
]

DIVERGING_COOLWARM = [
    "#3B4CC0",
    "#5977E3",
    "#7D9EF2",
    "#A2C0F9",
    "#C7DDFA",
    "#E8F0FC",
    "#F9EBE5",
    "#F6CFBB",
    "#F0AD8E",
    "#E68462",
    "#D8583E",
    "#C52A1E",
    "#B40426",
]


def select_optimal_palette(
    n_colors: int,
    *,
    palette_type: Literal["sequential", "diverging", "qualitative"] | None = None,
    data_range: tuple[float, float] | None = None,
    colorblind_safe: bool = True,
    background_color: str = "#FFFFFF",
    min_contrast_ratio: float = 4.5,
) -> list[str]:
    """Select optimal color palette based on data characteristics.

    : Automatically select optimal color palettes based on
    data characteristics, plot type, and accessibility requirements.

    Args:
        n_colors: Number of colors needed
        palette_type: Type of palette ("sequential", "diverging", "qualitative")
                     If None, auto-select based on n_colors and data_range
        data_range: Data range (min, max) for auto-detecting bipolar signals
        colorblind_safe: Ensure colorblind-safe palette (default: True)
        background_color: Background color for contrast checking (default: white)
        min_contrast_ratio: Minimum WCAG contrast ratio (default: 4.5 for AA)

    Returns:
        List of color hex codes

    Raises:
        ValueError: If n_colors is invalid or palette cannot meet requirements

    Example:
        >>> # Auto-select for 3 channels
        >>> colors = select_optimal_palette(3)
        >>> # Diverging palette for bipolar data
        >>> colors = select_optimal_palette(10, palette_type="diverging")

    References:
        VIS-023: Data-Driven Color Palette
        WCAG 2.1 contrast ratio guidelines (AA: 4.5:1, AAA: 7:1)
        ColorBrewer sequential/diverging schemes
    """
    if n_colors < 1:
        raise ValueError("n_colors must be >= 1")
    if min_contrast_ratio < 1.0:
        raise ValueError("min_contrast_ratio must be >= 1.0")

    # Auto-select palette type if not specified
    if palette_type is None:
        palette_type = _auto_select_palette_type(n_colors, data_range)

    # Select base palette
    if palette_type == "qualitative":
        base_colors = (
            COLORBLIND_SAFE_QUALITATIVE if colorblind_safe else _generate_qualitative(n_colors)
        )
    elif palette_type == "sequential":
        base_colors = SEQUENTIAL_VIRIDIS
    elif palette_type == "diverging":
        base_colors = DIVERGING_COOLWARM
    else:
        raise ValueError(f"Unknown palette_type: {palette_type}")

    # Sample colors if we need fewer than available
    if n_colors <= len(base_colors):
        # Evenly sample from palette
        indices = np.linspace(0, len(base_colors) - 1, n_colors).astype(int)
        colors = [base_colors[i] for i in indices]
    else:
        # Interpolate if we need more colors
        colors = _interpolate_colors(base_colors, n_colors)

    # Check contrast ratios
    colors_with_contrast = []
    bg_luminance = _relative_luminance(background_color)

    for color in colors:
        color_luminance = _relative_luminance(color)
        contrast = _contrast_ratio(color_luminance, bg_luminance)

        if contrast >= min_contrast_ratio:
            colors_with_contrast.append(color)
        else:
            # Adjust lightness to meet contrast requirement
            adjusted = _adjust_for_contrast(color, background_color, min_contrast_ratio)
            colors_with_contrast.append(adjusted)

    return colors_with_contrast


def _auto_select_palette_type(
    n_colors: int,
    data_range: tuple[float, float] | None,
) -> Literal["sequential", "diverging", "qualitative"]:
    """Auto-select palette type based on data characteristics.

    Args:
        n_colors: Number of colors needed
        data_range: Data range (min, max)

    Returns:
        Palette type
    """
    # Check for bipolar data (zero-crossing)
    if data_range is not None:
        min_val, max_val = data_range
        if min_val < 0 and max_val > 0:
            # Bipolar signal - use diverging
            return "diverging"

    # Multi-channel (distinct categories)
    if n_colors <= 8:
        return "qualitative"

    # Many colors or continuous data
    return "sequential"


def _relative_luminance(color: str) -> float:
    """Calculate relative luminance per WCAG 2.1.

    Args:
        color: Hex color code

    Returns:
        Relative luminance (0-1)
    """
    # Parse hex color
    color = color.removeprefix("#")

    r = int(color[0:2], 16) / 255.0
    g = int(color[2:4], 16) / 255.0
    b = int(color[4:6], 16) / 255.0

    # Convert to linear RGB
    def to_linear(c: float) -> float:
        if c <= 0.03928:
            return c / 12.92
        else:
            return ((c + 0.055) / 1.055) ** 2.4  # type: ignore[no-any-return]

    r_linear = to_linear(r)
    g_linear = to_linear(g)
    b_linear = to_linear(b)

    # Calculate luminance
    return 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear


def _contrast_ratio(lum1: float, lum2: float) -> float:
    """Calculate WCAG contrast ratio between two luminances.

    Args:
        lum1: First luminance (0-1)
        lum2: Second luminance (0-1)

    Returns:
        Contrast ratio (1-21)
    """
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)

    return (lighter + 0.05) / (darker + 0.05)


def _adjust_for_contrast(
    color: str,
    background: str,
    target_ratio: float,
) -> str:
    """Adjust color lightness to meet contrast requirement.

    Args:
        color: Color to adjust
        background: Background color
        target_ratio: Target contrast ratio

    Returns:
        Adjusted color hex code
    """
    # Parse color
    color_val = color.removeprefix("#")

    r = int(color_val[0:2], 16)
    g = int(color_val[2:4], 16)
    b = int(color_val[4:6], 16)

    # Convert to HSL for easier lightness adjustment
    h, s, l = _rgb_to_hsl(r, g, b)

    bg_lum = _relative_luminance(background)

    # Binary search for appropriate lightness
    l_min, l_max = 0.0, 1.0
    iterations = 0
    max_iterations = 20

    while iterations < max_iterations:
        # Try current lightness
        test_r, test_g, test_b = _hsl_to_rgb(h, s, l)
        test_color = f"#{test_r:02x}{test_g:02x}{test_b:02x}"
        test_lum = _relative_luminance(test_color)
        ratio = _contrast_ratio(test_lum, bg_lum)

        if abs(ratio - target_ratio) < 0.1:
            break

        if ratio < target_ratio:
            # Need more contrast - adjust lightness
            if bg_lum > 0.5:
                # Dark background - make lighter
                l_min = l
                l = (l + l_max) / 2
            else:
                # Light background - make darker
                l_max = l
                l = (l_min + l) / 2
        # Too much contrast - move back
        elif bg_lum > 0.5:
            l_max = l
            l = (l_min + l) / 2
        else:
            l_min = l
            l = (l + l_max) / 2

        iterations += 1

    final_r, final_g, final_b = _hsl_to_rgb(h, s, l)
    return f"#{final_r:02x}{final_g:02x}{final_b:02x}"


def _rgb_to_hsl(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Convert RGB to HSL color space.

    Args:
        r: Red value (0-255).
        g: Green value (0-255).
        b: Blue value (0-255).

    Returns:
        (h, s, l) tuple where h in [0, 360), s and l in [0, 1]
    """
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0

    max_c = max(r_norm, g_norm, b_norm)
    min_c = min(r_norm, g_norm, b_norm)
    delta = max_c - min_c

    # Lightness
    l = (max_c + min_c) / 2.0

    if delta == 0:
        # Achromatic
        return (0.0, 0.0, l)

    # Saturation
    s = delta / (max_c + min_c) if l < 0.5 else delta / (2.0 - max_c - min_c)

    # Hue
    if max_c == r_norm:
        h = ((g_norm - b_norm) / delta) % 6
    elif max_c == g_norm:
        h = ((b_norm - r_norm) / delta) + 2
    else:
        h = ((r_norm - g_norm) / delta) + 4

    h = h * 60.0

    return (h, s, l)


def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    """Convert HSL to RGB color space.

    Args:
        h: Hue in [0, 360)
        s: Saturation in [0, 1]
        l: Lightness in [0, 1]

    Returns:
        (r, g, b) tuple with values in [0, 255]
    """
    if s == 0:
        # Achromatic
        gray = int(l * 255)
        return (gray, gray, gray)

    def hue_to_rgb(p: float, q: float, t: float) -> float:
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    q = l * (1 + s) if l < 0.5 else l + s - l * s

    p = 2 * l - q

    h_norm = h / 360.0

    r = hue_to_rgb(p, q, h_norm + 1 / 3)
    g = hue_to_rgb(p, q, h_norm)
    b = hue_to_rgb(p, q, h_norm - 1 / 3)

    return (int(r * 255), int(g * 255), int(b * 255))


def _generate_qualitative(n_colors: int) -> list[str]:
    """Generate qualitative color palette.

    Args:
        n_colors: Number of colors

    Returns:
        List of hex color codes
    """
    # Generate evenly spaced hues
    colors = []
    for i in range(n_colors):
        hue = (i * 360.0 / n_colors) % 360
        r, g, b = _hsl_to_rgb(hue, 0.7, 0.5)
        colors.append(f"#{r:02x}{g:02x}{b:02x}")

    return colors


def _interpolate_colors(base_colors: list[str], n_colors: int) -> list[str]:
    """Interpolate between base colors to generate more colors.

    Args:
        base_colors: Base color palette
        n_colors: Target number of colors

    Returns:
        List of interpolated hex color codes
    """
    if n_colors <= len(base_colors):
        return base_colors[:n_colors]

    # Convert to RGB arrays
    rgb_array = np.zeros((len(base_colors), 3))
    for i, color in enumerate(base_colors):
        color = color.removeprefix("#")
        rgb_array[i, 0] = int(color[0:2], 16)
        rgb_array[i, 1] = int(color[2:4], 16)
        rgb_array[i, 2] = int(color[4:6], 16)

    # Interpolate
    indices = np.linspace(0, len(base_colors) - 1, n_colors)
    interp_rgb = np.zeros((n_colors, 3))

    for channel in range(3):
        interp_rgb[:, channel] = np.interp(
            indices, np.arange(len(base_colors)), rgb_array[:, channel]
        )

    # Convert back to hex
    colors = []
    for i in range(n_colors):
        r = int(interp_rgb[i, 0])
        g = int(interp_rgb[i, 1])
        b = int(interp_rgb[i, 2])
        colors.append(f"#{r:02x}{g:02x}{b:02x}")

    return colors


__all__ = [
    "COLORBLIND_SAFE_QUALITATIVE",
    "DIVERGING_COOLWARM",
    "SEQUENTIAL_VIRIDIS",
    "select_optimal_palette",
]
