"""Visualization rendering functions for DPI-aware output.

This module provides DPI-aware rendering configuration for adapting
plot quality and parameters based on target output device (screen vs print).


Example:
    >>> from oscura.visualization.render import configure_dpi_rendering
    >>> config = configure_dpi_rendering("publication")
    >>> fig = plt.figure(dpi=config['dpi'], figsize=config['figsize'])

References:
    - matplotlib DPI scaling best practices
    - Print quality standards (300-600 DPI)
"""

from __future__ import annotations

from typing import Any, Literal

RenderPreset = Literal["screen", "print", "publication"]


def configure_dpi_rendering(
    preset: RenderPreset = "screen",
    *,
    custom_dpi: int | None = None,
    dpi: int | None = None,
    figsize: tuple[float, float] = (10, 6),
    baseline_dpi: float = 96.0,
) -> dict[str, Any]:
    """Configure DPI-aware rendering parameters.

    Adapts plot rendering quality and parameters based on target DPI
    for print (300-600 DPI) versus screen (72-96 DPI) with export presets.

    Args:
        preset: Rendering preset ("screen", "print", "publication").
        custom_dpi: Custom DPI override (ignores preset).
        dpi: Alias for custom_dpi.
        figsize: Figure size in inches (width, height).
        baseline_dpi: Baseline DPI for scaling calculations (default 96).

    Returns:
        Dictionary with rendering configuration:
            - dpi: Target DPI
            - figsize: Figure size
            - font_scale: Font size scale factor
            - line_scale: Line width scale factor
            - marker_scale: Marker size scale factor
            - antialias: Whether to enable anti-aliasing
            - format: Recommended file format
            - style_params: Additional matplotlib rcParams

    Raises:
        ValueError: If preset is invalid.

    Example:
        >>> config = configure_dpi_rendering("print")
        >>> plt.rcParams.update(config['style_params'])
        >>> fig = plt.figure(dpi=config['dpi'], figsize=config['figsize'])

    References:
        VIS-017: DPI-Aware Rendering
    """
    presets = _get_dpi_presets()

    # Handle dpi alias and resolve preset
    if dpi is not None and custom_dpi is None:
        custom_dpi = dpi

    if preset not in presets and custom_dpi is None:
        raise ValueError(f"Invalid preset: {preset}. Must be one of {list(presets.keys())}")

    # Get configuration and DPI
    target_dpi, preset_config = _resolve_dpi_config(preset, custom_dpi, presets)

    # Calculate scale factors
    scale = target_dpi / baseline_dpi
    font_scale = scale
    line_scale = scale
    marker_scale = scale

    # Build style parameters
    style_params = _build_style_params(
        target_dpi, font_scale, line_scale, marker_scale, preset_config
    )

    _apply_antialias_settings(style_params, preset_config)

    if preset == "publication":
        _apply_publication_settings(style_params)

    return {
        "dpi": target_dpi,
        "figsize": figsize,
        "font_scale": font_scale,
        "line_scale": line_scale,
        "marker_scale": marker_scale,
        "antialias": preset_config["antialias"],
        "format": preset_config["format"],
        "style_params": style_params,
        "description": preset_config["description"],
        "preset": preset if custom_dpi is None else "custom",
    }


def _get_dpi_presets() -> dict[str, dict[str, Any]]:
    """Get DPI preset configurations."""
    return {
        "screen": {
            "dpi": 96,
            "font_family": "sans-serif",
            "antialias": True,
            "format": "png",
            "description": "Screen display (96 DPI)",
        },
        "print": {
            "dpi": 300,
            "font_family": "sans-serif",
            "antialias": False,
            "format": "pdf",
            "description": "Print output (300 DPI)",
        },
        "publication": {
            "dpi": 600,
            "font_family": "serif",
            "antialias": False,
            "format": "pdf",
            "description": "Publication quality (600 DPI)",
        },
    }


def _resolve_dpi_config(
    preset: str, custom_dpi: int | None, presets: dict[str, dict[str, Any]]
) -> tuple[int, dict[str, Any]]:
    """Resolve target DPI and configuration from preset or custom value."""
    if custom_dpi is not None:
        return custom_dpi, {
            "font_family": "sans-serif",
            "antialias": True,
            "format": "png" if custom_dpi <= 150 else "pdf",
            "description": f"Custom ({custom_dpi} DPI)",
        }

    preset_config = presets[preset]
    return preset_config["dpi"], preset_config


def _build_style_params(
    target_dpi: int,
    font_scale: float,
    line_scale: float,
    marker_scale: float,
    preset_config: dict[str, Any],
) -> dict[str, Any]:
    """Build matplotlib rcParams dictionary."""
    return {
        "figure.dpi": target_dpi,
        "savefig.dpi": target_dpi,
        "font.family": preset_config["font_family"],
        "font.size": 10 * font_scale,
        "axes.titlesize": 12 * font_scale,
        "axes.labelsize": 10 * font_scale,
        "xtick.labelsize": 9 * font_scale,
        "ytick.labelsize": 9 * font_scale,
        "legend.fontsize": 9 * font_scale,
        "lines.linewidth": 1.0 * line_scale,
        "lines.markersize": 6.0 * marker_scale,
        "patch.linewidth": 1.0 * line_scale,
        "grid.linewidth": 0.5 * line_scale,
        "axes.linewidth": 0.8 * line_scale,
        "xtick.major.width": 0.8 * line_scale,
        "ytick.major.width": 0.8 * line_scale,
        "xtick.minor.width": 0.6 * line_scale,
        "ytick.minor.width": 0.6 * line_scale,
    }


def _apply_antialias_settings(style_params: dict[str, Any], preset_config: dict[str, Any]) -> None:
    """Apply anti-aliasing settings to style params (modifies in-place)."""
    antialias = preset_config["antialias"]
    style_params["lines.antialiased"] = antialias
    style_params["patch.antialiased"] = antialias
    style_params["text.antialiased"] = antialias


def _apply_publication_settings(style_params: dict[str, Any]) -> None:
    """Apply publication-specific settings to style params (modifies in-place)."""
    style_params["font.family"] = "serif"
    style_params["mathtext.fontset"] = "cm"  # Computer Modern for LaTeX
    style_params["axes.grid"] = True
    style_params["grid.alpha"] = 0.3
    style_params["axes.axisbelow"] = True


def apply_rendering_config(config: dict[str, Any]) -> None:
    """Apply rendering configuration to matplotlib rcParams.

    Args:
        config: Configuration dictionary from configure_dpi_rendering().

    Raises:
        ImportError: If matplotlib is not available.

    Example:
        >>> config = configure_dpi_rendering("print")
        >>> apply_rendering_config(config)
    """
    try:
        import matplotlib.pyplot as plt

        plt.rcParams.update(config["style_params"])
    except ImportError:
        raise ImportError("matplotlib is required for rendering configuration")


__all__ = [
    "RenderPreset",
    "apply_rendering_config",
    "configure_dpi_rendering",
]
