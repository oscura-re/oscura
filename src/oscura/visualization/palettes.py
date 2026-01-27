"""Colorblind-safe palettes for Oscura visualizations.

This module provides colorblind-safe color palettes and utilities for
creating accessible visualizations.


Example:
    >>> from oscura.visualization.palettes import get_palette, show_palette
    >>> colors = get_palette("colorblind_safe")
    >>> show_palette("viridis")

References:
    - Wong, B. (2011). Color blindness. Nature Methods, 8(6), 441.
    - Colorbrewer 2.0 (colorbrewer2.org)
    - WCAG 2.1 color contrast guidelines
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from matplotlib.colors import Colormap

# Define colorblind-safe palettes
# Based on Wong 2011 and Paul Tol's schemes
PALETTES = {
    "colorblind_safe": [
        "#0173B2",  # Blue
        "#DE8F05",  # Orange
        "#029E73",  # Green
        "#CC78BC",  # Purple
        "#CA9161",  # Brown
        "#FBAFE4",  # Pink
        "#949494",  # Gray
        "#ECE133",  # Yellow
    ],
    "colorblind8": [  # Paul Tol's bright scheme
        "#4477AA",  # Blue
        "#EE6677",  # Red
        "#228833",  # Green
        "#CCBB44",  # Yellow
        "#66CCEE",  # Cyan
        "#AA3377",  # Purple
        "#BBBBBB",  # Gray
        "#EE8866",  # Orange
    ],
    "high_contrast": [
        "#000000",  # Black
        "#E69F00",  # Orange
        "#56B4E9",  # Sky Blue
        "#009E73",  # Bluish Green
        "#F0E442",  # Yellow
        "#0072B2",  # Blue
        "#D55E00",  # Vermillion
        "#CC79A7",  # Reddish Purple
    ],
    "grayscale": [
        "#000000",  # Black
        "#404040",  # Dark Gray
        "#808080",  # Medium Gray
        "#BFBFBF",  # Light Gray
        "#E0E0E0",  # Very Light Gray
    ],
}

# Line styles for multi-line plots
LINE_STYLES = ["solid", "dashed", "dotted", "dashdot"]

# Marker styles for scatter plots
MARKER_STYLES = ["o", "s", "^", "D", "v", "p", "*", "h"]


def get_palette(
    name: Literal[
        "colorblind_safe", "colorblind8", "high_contrast", "grayscale"
    ] = "colorblind_safe",
) -> list[str]:
    """Get a colorblind-safe color palette.

    : Default palette is colorblind-safe.
    Returns list of hex color codes that are distinguishable for colorblind users.

    Args:
        name: Palette name. Options:
            - "colorblind_safe": Default palette (Wong 2011) with 8 colors
            - "colorblind8": Paul Tol's bright scheme with 8 colors
            - "high_contrast": High contrast palette suitable for presentations
            - "grayscale": Grayscale palette for printing

    Returns:
        List of hex color codes

    Raises:
        ValueError: If palette name is not recognized

    Example:
        >>> from oscura.visualization.palettes import get_palette
        >>> colors = get_palette("colorblind_safe")
        >>> print(colors[0])  # First color (blue)
        #0173B2

    References:
        ACC-001: Colorblind-Safe Visualization Palette
    """
    if name not in PALETTES:
        valid = ", ".join(PALETTES.keys())
        raise ValueError(f"Unknown palette: {name}. Valid options: {valid}")
    return PALETTES[name].copy()


def get_colormap(
    name: Literal["viridis", "cividis", "plasma", "inferno", "magma"] = "viridis",
) -> Colormap:
    """Get a colorblind-safe matplotlib colormap.

    : Default palette is colorblind-safe (e.g., viridis, cividis).
    Returns perceptually uniform colormaps suitable for continuous data.

    Args:
        name: Colormap name. All options are colorblind-safe:
            - "viridis": Default, blue-green-yellow (recommended)
            - "cividis": Blue-yellow, optimized for CVD
            - "plasma": Purple-red-yellow
            - "inferno": Black-purple-yellow
            - "magma": Black-purple-white

    Returns:
        Matplotlib Colormap object

    Raises:
        ValueError: If colormap name is not recognized

    Example:
        >>> from oscura.visualization.palettes import get_colormap
        >>> import matplotlib.pyplot as plt
        >>> cmap = get_colormap("viridis")
        >>> plt.imshow(data, cmap=cmap)

    References:
        ACC-001: Colorblind-Safe Visualization Palette
    """
    valid = ["viridis", "cividis", "plasma", "inferno", "magma"]
    if name not in valid:
        raise ValueError(f"Unknown colormap: {name}. Valid options: {', '.join(valid)}")
    return plt.get_cmap(name)


def get_line_styles(
    n_lines: int,
    *,
    palette: str = "colorblind_safe",
    cycle_styles: bool = True,
) -> list[dict]:  # type: ignore[type-arg]
    """Get line styles for multi-line plots.

    : Multi-line plots use distinct line styles in addition to colors.
    Combines colors with line styles for maximum distinguishability.

    Args:
        n_lines: Number of lines to style
        palette: Palette name for colors
        cycle_styles: Cycle through line styles if more lines than styles

    Returns:
        List of style dictionaries with 'color' and 'linestyle' keys

    Example:
        >>> from oscura.visualization.palettes import get_line_styles
        >>> styles = get_line_styles(4)
        >>> for i, style in enumerate(styles):
        ...     plt.plot(x, y[i], **style, label=f"Line {i}")

    References:
        ACC-001: Multi-line plots use distinct line styles in addition to colors
    """
    colors = get_palette(palette)  # type: ignore[arg-type]
    styles = []

    for i in range(n_lines):
        color = colors[i % len(colors)]
        linestyle = LINE_STYLES[i % len(LINE_STYLES)] if cycle_styles else "solid"

        styles.append({"color": color, "linestyle": linestyle})

    return styles


def get_pass_fail_symbols() -> dict[str, str]:
    """Get pass/fail symbols for accessible reporting.

    : Pass/fail uses symbols (✓/✗) not just red/green.
    Returns symbols that work in text and don't rely on color alone.

    Returns:
        Dictionary with 'pass' and 'fail' symbol keys

    Example:
        >>> from oscura.visualization.palettes import get_pass_fail_symbols
        >>> symbols = get_pass_fail_symbols()
        >>> print(f"{symbols['pass']} Test passed")
        ✓ Test passed

    References:
        ACC-001: Pass/fail uses symbols (✓/✗) not just red/green
    """
    return {"pass": "✓", "fail": "✗"}


def get_pass_fail_colors(
    *,
    colorblind_safe: bool = True,
) -> dict[str, str]:
    """Get pass/fail colors.

    : Pass/fail colors are colorblind-safe when combined with symbols.
    Returns green/red or blue/orange based on colorblind_safe flag.

    Args:
        colorblind_safe: Use colorblind-safe blue/orange instead of green/red

    Returns:
        Dictionary with 'pass' and 'fail' color keys (hex codes)

    Example:
        >>> from oscura.visualization.palettes import get_pass_fail_colors
        >>> colors = get_pass_fail_colors(colorblind_safe=True)
        >>> print(colors['pass'])  # Blue for pass
        #0173B2

    References:
        ACC-001: Colorblind-Safe Visualization Palette
    """
    if colorblind_safe:
        return {
            "pass": "#0173B2",  # Blue
            "fail": "#DE8F05",  # Orange
        }
    else:
        return {
            "pass": "#2CA02C",  # Green
            "fail": "#D62728",  # Red
        }


def show_palette(
    name: str = "colorblind_safe",
    *,
    save_path: str | None = None,
) -> None:
    """Display a color palette preview.

    : Palette preview: show_palette(name).
    Shows palette colors in a matplotlib figure for visual inspection.

    Args:
        name: Palette name or colormap name
        save_path: Optional path to save the figure

    Raises:
        ValueError: If unknown palette or colormap name.

    Example:
        >>> from oscura.visualization.palettes import show_palette
        >>> show_palette("colorblind_safe")
        >>> show_palette("viridis", save_path="palette.png")

    References:
        ACC-001: Colorblind-Safe Visualization Palette
    """
    _fig, ax = plt.subplots(figsize=(8, 2))

    # Check if it's a discrete palette or continuous colormap
    if name in PALETTES:
        # Discrete palette
        colors = PALETTES[name]
        n_colors = len(colors)
        x = np.arange(n_colors)

        # Create color swatches
        for i, color in enumerate(colors):
            ax.add_patch(Rectangle((i, 0), 1, 1, facecolor=color, edgecolor="black"))

        ax.set_xlim(0, n_colors)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks(x + 0.5)
        ax.set_xticklabels([f"C{i}" for i in range(n_colors)])
        ax.set_title(f"Palette: {name}")

    else:
        # Continuous colormap
        try:
            cmap = get_colormap(name)  # type: ignore[arg-type]
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            ax.imshow(gradient, aspect="auto", cmap=cmap)
            ax.set_yticks([])
            ax.set_xticks([0, 64, 128, 192, 255])
            ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"])
            ax.set_title(f"Colormap: {name}")
        except ValueError:
            raise ValueError(f"Unknown palette or colormap: {name}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def create_custom_palette(
    colors: list[str],
    *,
    name: str = "custom",
) -> list[str]:
    """Create a custom color palette.

    : Custom palette creation support.
    Validates and registers a custom color palette.

    Args:
        colors: List of hex color codes
        name: Name for the custom palette

    Returns:
        List of validated hex color codes

    Raises:
        ValueError: If color codes are invalid

    Example:
        >>> from oscura.visualization.palettes import create_custom_palette
        >>> custom = create_custom_palette(
        ...     ["#FF0000", "#00FF00", "#0000FF"],
        ...     name="rgb"
        ... )
        >>> print(custom)
        ['#FF0000', '#00FF00', '#0000FF']

    References:
        ACC-001: Custom palette creation support
    """
    # Validate hex codes
    import re

    hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
    validated = []

    for color in colors:
        if not hex_pattern.match(color):
            raise ValueError(f"Invalid hex color code: {color}")
        validated.append(color.upper())

    # Optionally register the palette
    PALETTES[name] = validated

    return validated


def simulate_colorblindness(
    color: str,
    *,
    deficiency: Literal["protanopia", "deuteranopia", "tritanopia"] = "deuteranopia",
) -> str:
    """Simulate how a color appears with color vision deficiency.

    : Test with color blindness simulators.
    Converts a color to approximate how it appears with CVD.

    Args:
        color: Hex color code
        deficiency: Type of color vision deficiency:
            - "protanopia": Red-blind (1% of males)
            - "deuteranopia": Green-blind (1% of males)
            - "tritanopia": Blue-blind (rare)

    Returns:
        Simulated hex color code

    Raises:
        ValueError: If unknown deficiency type.

    Example:
        >>> from oscura.visualization.palettes import simulate_colorblindness
        >>> red = "#FF0000"
        >>> simulated = simulate_colorblindness(red, deficiency="deuteranopia")
        >>> print(simulated)  # Appears brownish
        #9C7A00

    References:
        ACC-001: Test with color blindness simulators
        Brettel, H., Viénot, F., & Mollon, J. D. (1997). Computerized simulation
        of color appearance for dichromats. JOSA A, 14(10), 2647-2655.
    """
    # Convert hex to RGB
    hex_color = color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

    # Normalize to 0-1
    r, g, b = r / 255.0, g / 255.0, b / 255.0  # type: ignore[assignment]

    # Apply transformation matrices (simplified Brettel algorithm)
    if deficiency == "protanopia":
        # Red-blind: confuse red and green
        r_sim = 0.56667 * r + 0.43333 * g
        g_sim = 0.55833 * r + 0.44167 * g
        b_sim = b
    elif deficiency == "deuteranopia":
        # Green-blind: confuse red and green
        r_sim = 0.625 * r + 0.375 * g
        g_sim = 0.7 * r + 0.3 * g
        b_sim = b
    elif deficiency == "tritanopia":
        # Blue-blind: confuse blue and yellow
        r_sim = r
        g_sim = 0.95 * g + 0.05 * b
        b_sim = 0.433 * g + 0.567 * b  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown deficiency: {deficiency}")

    # Convert back to 0-255 and hex
    r_int = int(np.clip(r_sim * 255, 0, 255))
    g_int = int(np.clip(g_sim * 255, 0, 255))
    b_int = int(np.clip(b_sim * 255, 0, 255))

    return f"#{r_int:02X}{g_int:02X}{b_int:02X}"


__all__ = [
    "LINE_STYLES",
    "MARKER_STYLES",
    "PALETTES",
    "create_custom_palette",
    "get_colormap",
    "get_line_styles",
    "get_palette",
    "get_pass_fail_colors",
    "get_pass_fail_symbols",
    "show_palette",
    "simulate_colorblindness",
]
