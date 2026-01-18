"""Plot style presets for different output contexts.

This module provides comprehensive style presets for publication-quality,
presentation, screen viewing, and print output.


Example:
    >>> from oscura.visualization.styles import apply_style_preset
    >>> with apply_style_preset("publication"):
    ...     plot_waveform(signal)

References:
    matplotlib rcParams customization
    Publication and presentation best practices
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class StylePreset:
    """Style preset configuration for plots.

    Attributes:
        name: Preset name
        dpi: Target DPI (dots per inch)
        font_family: Font family (serif, sans-serif, monospace)
        font_size: Base font size in points
        line_width: Default line width in points
        marker_size: Default marker size
        figure_facecolor: Figure background color
        axes_facecolor: Axes background color
        axes_edgecolor: Axes edge color
        grid_color: Grid line color
        grid_alpha: Grid line transparency
        grid_linestyle: Grid line style
        use_latex: Use LaTeX for text rendering
        tight_layout: Use tight layout
        rcparams: Additional matplotlib rcParams
    """

    name: str
    dpi: int = 96
    font_family: str = "sans-serif"
    font_size: int = 10
    line_width: float = 1.0
    marker_size: float = 6.0
    figure_facecolor: str = "white"
    axes_facecolor: str = "white"
    axes_edgecolor: str = "black"
    grid_color: str = "#B0B0B0"
    grid_alpha: float = 0.3
    grid_linestyle: str = "-"
    use_latex: bool = False
    tight_layout: bool = True
    rcparams: dict[str, Any] = field(default_factory=dict)


# Predefined style presets

PUBLICATION_PRESET = StylePreset(
    name="publication",
    dpi=600,
    font_family="serif",
    font_size=10,
    line_width=0.8,
    marker_size=4.0,
    figure_facecolor="white",
    axes_facecolor="white",
    axes_edgecolor="black",
    grid_color="#808080",
    grid_alpha=0.3,
    grid_linestyle=":",
    use_latex=False,  # LaTeX optional - requires system install
    tight_layout=True,
    rcparams={
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "lines.antialiased": True,
        "patch.antialiased": True,
        "savefig.dpi": 600,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
    },
)

PRESENTATION_PRESET = StylePreset(
    name="presentation",
    dpi=96,
    font_family="sans-serif",
    font_size=18,
    line_width=2.5,
    marker_size=10.0,
    figure_facecolor="white",
    axes_facecolor="white",
    axes_edgecolor="black",
    grid_color="#CCCCCC",
    grid_alpha=0.5,
    grid_linestyle="-",
    use_latex=False,
    tight_layout=True,
    rcparams={
        "axes.linewidth": 2.0,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "lines.antialiased": True,
        "savefig.dpi": 150,
    },
)

SCREEN_PRESET = StylePreset(
    name="screen",
    dpi=96,
    font_family="sans-serif",
    font_size=10,
    line_width=1.2,
    marker_size=6.0,
    figure_facecolor="white",
    axes_facecolor="white",
    axes_edgecolor="#333333",
    grid_color="#B0B0B0",
    grid_alpha=0.3,
    grid_linestyle="-",
    use_latex=False,
    tight_layout=True,
    rcparams={
        "axes.linewidth": 1.0,
        "lines.antialiased": True,
        "patch.antialiased": True,
    },
)

PRINT_PRESET = StylePreset(
    name="print",
    dpi=300,
    font_family="serif",
    font_size=11,
    line_width=1.2,
    marker_size=5.0,
    figure_facecolor="white",
    axes_facecolor="white",
    axes_edgecolor="black",
    grid_color="#707070",
    grid_alpha=0.3,
    grid_linestyle=":",
    use_latex=False,
    tight_layout=True,
    rcparams={
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "lines.antialiased": False,  # Sharper lines for print
        "patch.antialiased": False,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
    },
)

# Registry of available presets
PRESETS: dict[str, StylePreset] = {
    "publication": PUBLICATION_PRESET,
    "presentation": PRESENTATION_PRESET,
    "screen": SCREEN_PRESET,
    "print": PRINT_PRESET,
}


@contextmanager
def apply_style_preset(
    preset: str | StylePreset,
    *,
    overrides: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Apply style preset as context manager.

    : Provide comprehensive style presets for common use cases
    with support for custom overrides.

    Args:
        preset: Preset name or StylePreset object
        overrides: Dictionary of rcParams to override

    Yields:
        None (use as context manager)

    Raises:
        ValueError: If preset name is unknown
        ImportError: If matplotlib is not available

    Example:
        >>> with apply_style_preset("publication"):
        ...     fig, ax = plt.subplots()
        ...     ax.plot(x, y)
        ...     plt.savefig("figure.pdf")

        >>> # With overrides
        >>> with apply_style_preset("screen", overrides={"font.size": 14}):
        ...     plot_waveform(signal)

    References:
        VIS-024: Plot Style Presets
        matplotlib style sheets and rcParams
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for style presets")

    # Get preset object
    if isinstance(preset, str):
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
        preset_obj = PRESETS[preset]
    else:
        preset_obj = preset

    # Build rcParams dictionary
    rc_dict = _preset_to_rcparams(preset_obj)

    # Apply overrides
    if overrides:
        rc_dict.update(overrides)

    # Apply as context
    with plt.rc_context(rc_dict):
        yield


def _preset_to_rcparams(preset: StylePreset) -> dict[str, Any]:
    """Convert StylePreset to matplotlib rcParams dictionary.

    Args:
        preset: StylePreset object

    Returns:
        Dictionary of rcParams
    """
    rc = {
        "figure.dpi": preset.dpi,
        "font.family": preset.font_family,
        "font.size": preset.font_size,
        "lines.linewidth": preset.line_width,
        "lines.markersize": preset.marker_size,
        "figure.facecolor": preset.figure_facecolor,
        "axes.facecolor": preset.axes_facecolor,
        "axes.edgecolor": preset.axes_edgecolor,
        "grid.color": preset.grid_color,
        "grid.alpha": preset.grid_alpha,
        "grid.linestyle": preset.grid_linestyle,
        "figure.autolayout": preset.tight_layout,
    }

    # LaTeX rendering
    if preset.use_latex:
        rc["text.usetex"] = True

    # Merge with additional rcparams
    rc.update(preset.rcparams)

    return rc


def create_custom_preset(
    name: str,
    base_preset: str = "screen",
    **kwargs: Any,
) -> StylePreset:
    """Create custom preset by inheriting from base preset.

    : Support custom presets with inheritance and override.

    Args:
        name: Name for custom preset
        base_preset: Base preset to inherit from
        **kwargs: Attributes to override

    Returns:
        Custom StylePreset object

    Raises:
        ValueError: If base_preset is unknown

    Example:
        >>> custom = create_custom_preset(
        ...     "my_style",
        ...     base_preset="publication",
        ...     font_size=12,
        ...     line_width=1.5
        ... )
        >>> with apply_style_preset(custom):
        ...     plot_data()

    References:
        VIS-024: Plot Style Presets with inheritance
    """
    if base_preset not in PRESETS:
        raise ValueError(f"Unknown base_preset: {base_preset}")

    # Get base preset
    base = PRESETS[base_preset]

    # Create copy with overrides
    preset_dict = {
        "name": name,
        "dpi": kwargs.get("dpi", base.dpi),
        "font_family": kwargs.get("font_family", base.font_family),
        "font_size": kwargs.get("font_size", base.font_size),
        "line_width": kwargs.get("line_width", base.line_width),
        "marker_size": kwargs.get("marker_size", base.marker_size),
        "figure_facecolor": kwargs.get("figure_facecolor", base.figure_facecolor),
        "axes_facecolor": kwargs.get("axes_facecolor", base.axes_facecolor),
        "axes_edgecolor": kwargs.get("axes_edgecolor", base.axes_edgecolor),
        "grid_color": kwargs.get("grid_color", base.grid_color),
        "grid_alpha": kwargs.get("grid_alpha", base.grid_alpha),
        "grid_linestyle": kwargs.get("grid_linestyle", base.grid_linestyle),
        "use_latex": kwargs.get("use_latex", base.use_latex),
        "tight_layout": kwargs.get("tight_layout", base.tight_layout),
        "rcparams": kwargs.get("rcparams", base.rcparams.copy()),
    }

    return StylePreset(**preset_dict)


def register_preset(preset: StylePreset) -> None:
    """Register custom preset in global registry.

    Args:
        preset: StylePreset to register

    Example:
        >>> custom = create_custom_preset("my_style", base_preset="publication")
        >>> register_preset(custom)
        >>> with apply_style_preset("my_style"):
        ...     plot_data()
    """
    PRESETS[preset.name] = preset


def list_presets() -> list[str]:
    """Get list of available preset names.

    Returns:
        List of preset names

    Example:
        >>> presets = list_presets()
        >>> print(presets)
        ['publication', 'presentation', 'screen', 'print']
    """
    return list(PRESETS.keys())


__all__ = [
    "PRESENTATION_PRESET",
    "PRESETS",
    "PRINT_PRESET",
    "PUBLICATION_PRESET",
    "SCREEN_PRESET",
    "StylePreset",
    "apply_style_preset",
    "create_custom_preset",
    "list_presets",
    "register_preset",
]
