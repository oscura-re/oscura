"""Figure management for saving and organizing matplotlib figures.

This module provides utilities for saving matplotlib figures in multiple formats
and managing collections of figures for report generation.

Example:
    >>> from oscura.visualization.figure_manager import FigureManager
    >>> manager = FigureManager(output_dir="./plots")
    >>> paths = manager.save_figure(fig, "timing_diagram", formats=["png", "svg"])
    >>> base64_img = manager.embed_as_base64(fig, format="png")
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class FigureManager:
    """Manager for saving and organizing matplotlib figures.

    Attributes:
        output_dir: Directory for saving figures.
        saved_figures: Dictionary mapping figure names to saved paths.
    """

    def __init__(self, output_dir: str | Path):
        """Initialize figure manager.

        Args:
            output_dir: Directory for saving figures.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.saved_figures: dict[str, dict[str, Path]] = {}

    def save_figure(
        self,
        fig: Figure,
        name: str,
        *,
        formats: list[str] | None = None,
        dpi: int = 300,
        **savefig_kwargs: Any,
    ) -> dict[str, Path]:
        """Save figure in multiple formats.

        Args:
            fig: Matplotlib figure to save.
            name: Base name for the saved files (without extension).
            formats: List of formats to save ("png", "svg", "pdf"). Defaults to ["png"].
            dpi: Resolution for raster formats (default: 300).
            **savefig_kwargs: Additional kwargs passed to fig.savefig().

        Returns:
            Dictionary mapping format to saved file path.

        Example:
            >>> paths = manager.save_figure(fig, "timing_diagram", formats=["png", "svg"])
            >>> print(paths["png"])  # PosixPath('./plots/timing_diagram.png')
        """
        if formats is None:
            formats = ["png"]

        saved_paths: dict[str, Path] = {}

        for fmt in formats:
            # Construct file path
            file_path = self.output_dir / f"{name}.{fmt}"

            # Save figure
            fig.savefig(
                file_path,
                dpi=dpi,
                bbox_inches="tight",
                format=fmt,
                **savefig_kwargs,
            )

            saved_paths[fmt] = file_path

        # Store in saved_figures registry
        self.saved_figures[name] = saved_paths

        return saved_paths

    def embed_as_base64(
        self,
        fig: Figure,
        format: str = "png",
        dpi: int = 150,
        **savefig_kwargs: Any,
    ) -> str:
        """Convert figure to base64-encoded string for HTML embedding.

        Args:
            fig: Matplotlib figure to convert.
            format: Image format ("png", "jpg", "svg"). Default: "png".
            dpi: Resolution for raster formats (default: 150).
            **savefig_kwargs: Additional kwargs passed to fig.savefig().

        Returns:
            Base64-encoded image string (without data URI prefix).

        Example:
            >>> base64_img = manager.embed_as_base64(fig)
            >>> html = f'<img src="data:image/png;base64,{base64_img}" />'
        """
        # Save figure to bytes buffer
        buf = BytesIO()
        fig.savefig(
            buf,
            format=format,
            dpi=dpi,
            bbox_inches="tight",
            **savefig_kwargs,
        )
        buf.seek(0)

        # Encode to base64
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        return img_base64

    def get_saved_path(self, name: str, format: str) -> Path | None:
        """Get path to a saved figure.

        Args:
            name: Figure name.
            format: Image format.

        Returns:
            Path to saved figure, or None if not found.
        """
        if name in self.saved_figures:
            return self.saved_figures[name].get(format)
        return None

    def list_saved_figures(self) -> list[str]:
        """Get list of all saved figure names.

        Returns:
            List of figure names.
        """
        return list(self.saved_figures.keys())


__all__ = [
    "FigureManager",
]
