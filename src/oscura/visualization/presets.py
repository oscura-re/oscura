"""Comprehensive visualization presets combining styles, colors, and rendering.

This module provides integrated presets that combine style settings, color
palettes, and rendering configuration for different use cases.


Example:
    >>> from oscura.visualization.presets import apply_preset
    >>> with apply_preset("ieee_publication"):
    ...     plot_waveform(signal)

References:
    - IEEE publication standards
    - Presentation best practices
    - Colorblind-safe palette design
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class VisualizationPreset:
    """Complete visualization preset configuration.

    Attributes:
        name: Preset name
        description: Preset description
        style_params: Matplotlib rcParams
        color_palette: List of colors for multi-channel plots
        dpi: Target DPI
        figure_size: Default figure size (width, height) in inches
        font_family: Font family
        colorblind_safe: Whether palette is colorblind-safe
        print_optimized: Whether optimized for print output
    """

    name: str
    description: str
    style_params: dict[str, Any]
    color_palette: list[str]
    dpi: int = 96
    figure_size: tuple[float, float] = (10, 6)
    font_family: str = "sans-serif"
    colorblind_safe: bool = True
    print_optimized: bool = False


# IEEE Publication Preset (VIS-020)
IEEE_PUBLICATION_PRESET = VisualizationPreset(
    name="ieee_publication",
    description="IEEE publication quality (single-column, grayscale-friendly)",
    dpi=600,
    figure_size=(3.5, 2.5),  # IEEE single-column width
    font_family="serif",
    colorblind_safe=True,
    print_optimized=True,
    color_palette=[
        "#000000",  # Black
        "#555555",  # Dark gray
        "#AAAAAA",  # Light gray
        "#0173B2",  # Blue (grayscale-safe)
        "#DE8F05",  # Orange (grayscale-safe)
        "#029E73",  # Green (grayscale-safe)
    ],
    style_params={
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 0.8,
        "lines.markersize": 3.0,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "axes.grid": True,
        "axes.axisbelow": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "lines.antialiased": False,  # Sharper for print
        "patch.antialiased": False,
        "mathtext.fontset": "cm",  # Computer Modern (LaTeX-like)
    },
)

# IEEE Double-Column Preset
IEEE_DOUBLE_COLUMN_PRESET = VisualizationPreset(
    name="ieee_double_column",
    description="IEEE publication quality (double-column width)",
    dpi=600,
    figure_size=(7.0, 2.5),  # IEEE double-column width
    font_family="serif",
    colorblind_safe=True,
    print_optimized=True,
    color_palette=IEEE_PUBLICATION_PRESET.color_palette,
    style_params=IEEE_PUBLICATION_PRESET.style_params.copy(),
)

# Presentation Preset
PRESENTATION_PRESET = VisualizationPreset(
    name="presentation",
    description="Presentation slides (high contrast, large fonts, bold lines)",
    dpi=96,
    figure_size=(12, 7),
    font_family="sans-serif",
    colorblind_safe=True,
    print_optimized=False,
    color_palette=[
        "#0173B2",  # Blue
        "#DE8F05",  # Orange
        "#029E73",  # Green
        "#CC78BC",  # Purple
        "#CA9161",  # Brown
        "#ECE133",  # Yellow
    ],
    style_params={
        "figure.dpi": 96,
        "font.family": "sans-serif",
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "lines.linewidth": 3.0,
        "lines.markersize": 10.0,
        "axes.linewidth": 2.0,
        "grid.linewidth": 1.0,
        "grid.alpha": 0.4,
        "axes.grid": True,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "lines.antialiased": True,
        "patch.antialiased": True,
    },
)

# Screen/Interactive Preset
SCREEN_PRESET = VisualizationPreset(
    name="screen",
    description="Screen viewing (vibrant colors, medium fonts, anti-aliased)",
    dpi=96,
    figure_size=(10, 6),
    font_family="sans-serif",
    colorblind_safe=True,
    print_optimized=False,
    color_palette=[
        "#1F77B4",  # Blue
        "#FF7F0E",  # Orange
        "#2CA02C",  # Green
        "#D62728",  # Red
        "#9467BD",  # Purple
        "#8C564B",  # Brown
        "#E377C2",  # Pink
        "#7F7F7F",  # Gray
    ],
    style_params={
        "figure.dpi": 96,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "lines.markersize": 6.0,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.3,
        "axes.grid": True,
        "lines.antialiased": True,
        "patch.antialiased": True,
    },
)

# Print Preset
PRINT_PRESET = VisualizationPreset(
    name="print",
    description="Print output (300 DPI, CMYK-safe colors, optimized file size)",
    dpi=300,
    figure_size=(8, 5),
    font_family="serif",
    colorblind_safe=True,
    print_optimized=True,
    color_palette=[
        "#0173B2",  # Blue (CMYK-safe)
        "#DE8F05",  # Orange (CMYK-safe)
        "#029E73",  # Green (CMYK-safe)
        "#CC78BC",  # Purple (CMYK-safe)
        "#555555",  # Gray (CMYK-safe)
    ],
    style_params={
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.2,
        "lines.markersize": 5.0,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.3,
        "axes.grid": True,
        "lines.antialiased": False,  # Cleaner for print
        "patch.antialiased": False,
    },
)

# Dark Theme Preset
DARK_THEME_PRESET = VisualizationPreset(
    name="dark",
    description="Dark theme (dark background, high-contrast colors)",
    dpi=96,
    figure_size=(10, 6),
    font_family="sans-serif",
    colorblind_safe=True,
    print_optimized=False,
    color_palette=[
        "#56B4E9",  # Light blue
        "#E69F00",  # Orange
        "#009E73",  # Green
        "#F0E442",  # Yellow
        "#CC79A7",  # Pink
        "#0072B2",  # Blue
    ],
    style_params={
        "figure.dpi": 96,
        "figure.facecolor": "#1E1E1E",
        "axes.facecolor": "#2D2D2D",
        "axes.edgecolor": "#CCCCCC",
        "axes.labelcolor": "#CCCCCC",
        "text.color": "#CCCCCC",
        "xtick.color": "#CCCCCC",
        "ytick.color": "#CCCCCC",
        "grid.color": "#555555",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 10,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "lines.antialiased": True,
    },
)

# Preset registry
PRESETS: dict[str, VisualizationPreset] = {
    "ieee_publication": IEEE_PUBLICATION_PRESET,
    "ieee_double_column": IEEE_DOUBLE_COLUMN_PRESET,
    "presentation": PRESENTATION_PRESET,
    "screen": SCREEN_PRESET,
    "print": PRINT_PRESET,
    "dark": DARK_THEME_PRESET,
}


@contextmanager
def apply_preset(
    preset: str | VisualizationPreset,
    *,
    overrides: dict[str, Any] | None = None,
) -> Iterator[VisualizationPreset]:
    """Apply visualization preset as context manager./VIS-024.

    Combines style settings, color palette, and rendering configuration.

    Args:
        preset: Preset name or VisualizationPreset object.
        overrides: Dictionary of rcParams to override.

    Yields:
        VisualizationPreset object for access to color palette.

    Raises:
        ValueError: If preset name is unknown.
        ImportError: If matplotlib is not available.

    Example:
        >>> with apply_preset("ieee_publication") as preset:
        ...     fig, ax = plt.subplots(figsize=preset.figure_size)
        ...     ax.plot(x, y, color=preset.color_palette[0])
        ...     plt.savefig("figure.pdf")

        >>> # With custom overrides
        >>> with apply_preset("screen", overrides={"font.size": 14}):
        ...     plot_waveform(signal)

    References:
        VIS-020: IEEE Publication Style Preset
        VIS-024: Plot Style Presets
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization presets")

    # Get preset object
    if isinstance(preset, str):
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
        preset_obj = PRESETS[preset]
    else:
        preset_obj = preset

    # Build rcParams dictionary
    # NECESSARY COPY: rc_dict is modified by .update() below.
    # Copy prevents mutations from affecting original preset.
    rc_dict = preset_obj.style_params.copy()

    # Apply overrides
    if overrides:
        rc_dict.update(overrides)

    # Apply as context
    with plt.rc_context(rc_dict):
        yield preset_obj


def get_preset_colors(
    preset: str | VisualizationPreset,
    n_colors: int | None = None,
) -> list[str]:
    """Get color palette from preset.

    Args:
        preset: Preset name or object.
        n_colors: Number of colors to return (None = all).

    Returns:
        List of color hex codes.

    Raises:
        ValueError: If unknown preset name.

    Example:
        >>> colors = get_preset_colors("ieee_publication", n_colors=3)
        >>> # Use colors for multi-channel plot

    References:
        VIS-023: Data-Driven Color Palette
        VIS-024: Plot Style Presets
    """
    if isinstance(preset, str):
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        preset_obj = PRESETS[preset]
    else:
        preset_obj = preset

    colors = preset_obj.color_palette

    if n_colors is not None:
        if n_colors <= len(colors):
            return colors[:n_colors]
        else:
            # Cycle colors if more needed
            return [colors[i % len(colors)] for i in range(n_colors)]

    return colors


def list_presets() -> list[str]:
    """Get list of available preset names.

    Returns:
        List of preset names.

    Example:
        >>> presets = list_presets()
        >>> print(presets)
        ['ieee_publication', 'presentation', 'screen', 'print', 'dark']
    """
    return list(PRESETS.keys())


def create_custom_preset(
    name: str,
    base_preset: str = "screen",
    **kwargs: Any,
) -> VisualizationPreset:
    """Create custom preset by inheriting from base.

    Args:
        name: Name for custom preset.
        base_preset: Base preset to inherit from.
        **kwargs: Attributes to override.

    Returns:
        Custom VisualizationPreset object.

    Raises:
        ValueError: If base_preset is unknown.

    Example:
        >>> custom = create_custom_preset(
        ...     "my_preset",
        ...     base_preset="ieee_publication",
        ...     figure_size=(5, 3),
        ...     dpi=300,
        ... )
        >>> with apply_preset(custom):
        ...     plot_data()

    References:
        VIS-024: Plot Style Presets (custom preset creation)
    """
    if base_preset not in PRESETS:
        raise ValueError(f"Unknown base_preset: {base_preset}")

    base = PRESETS[base_preset]

    # Create copy with overrides
    preset_dict = {
        "name": name,
        "description": kwargs.get("description", f"Custom preset based on {base_preset}"),
        "style_params": kwargs.get("style_params", base.style_params.copy()),
        "color_palette": kwargs.get("color_palette", base.color_palette.copy()),
        "dpi": kwargs.get("dpi", base.dpi),
        "figure_size": kwargs.get("figure_size", base.figure_size),
        "font_family": kwargs.get("font_family", base.font_family),
        "colorblind_safe": kwargs.get("colorblind_safe", base.colorblind_safe),
        "print_optimized": kwargs.get("print_optimized", base.print_optimized),
    }

    return VisualizationPreset(**preset_dict)


__all__ = [
    "DARK_THEME_PRESET",
    "IEEE_DOUBLE_COLUMN_PRESET",
    "IEEE_PUBLICATION_PRESET",
    "PRESENTATION_PRESET",
    "PRESETS",
    "PRINT_PRESET",
    "SCREEN_PRESET",
    "VisualizationPreset",
    "apply_preset",
    "create_custom_preset",
    "get_preset_colors",
    "list_presets",
]
