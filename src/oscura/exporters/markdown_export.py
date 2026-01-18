"""Markdown report export for Oscura.

This module provides Markdown report generation with measurement tables,
plot references, and configurable sections.


Example:
    >>> from oscura.exporters.markdown_export import export_markdown
    >>> export_markdown(measurements, "report.md", title="Analysis Report")
"""

from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any


def export_markdown(
    data: dict[str, Any],
    path: str | Path,
    *,
    title: str = "Oscura Analysis Report",
    author: str | None = None,
    include_plots: bool = True,
    embed_images: bool = True,
    sections: list[str] | None = None,
) -> None:
    """Export measurement results to Markdown format.

    Args:
        data: Dictionary containing measurement results, plots, and metadata.
            Expected keys:
            - "measurements": dict of name -> value pairs
            - "plots": list of matplotlib figures or paths
            - "metadata": optional dict of metadata
            - "summary": optional executive summary text
        path: Output file path.
        title: Report title.
        author: Author name (optional).
        include_plots: Include plots in report.
        embed_images: Embed images as base64 (True) or save separately (False).
        sections: List of sections to include. If None, includes all available.
            Options: "metadata", "summary", "measurements", "plots", "conclusions"

    References:
        EXP-006
    """
    lines: list[str] = []

    # Header
    lines.append(f"# {title}\n")
    lines.append("")

    # Metadata section
    if sections is None or "metadata" in sections:
        lines.extend(_generate_metadata_section(data, author))

    # Executive summary
    if (sections is None or "summary" in sections) and "summary" in data:
        lines.append("## Executive Summary\n")
        lines.append(data["summary"])
        lines.append("")

    # Measurements table
    if (sections is None or "measurements" in sections) and "measurements" in data:
        lines.extend(_generate_measurements_section(data["measurements"]))

    # Plots
    if include_plots and (sections is None or "plots" in sections) and "plots" in data:
        lines.extend(_generate_plots_section(data["plots"], path, embed_images))

    # Conclusions
    if (sections is None or "conclusions" in sections) and "conclusions" in data:
        lines.append("## Conclusions\n")
        lines.append(data["conclusions"])
        lines.append("")

    # Write to file
    content = "\n".join(lines)
    Path(path).write_text(content, encoding="utf-8")


def _generate_metadata_section(data: dict[str, Any], author: str | None) -> list[str]:
    """Generate metadata section."""
    lines = ["## Report Information\n", ""]

    metadata = data.get("metadata", {})

    lines.append(f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if author:
        lines.append(f"- **Author**: {author}")

    if "filename" in metadata:
        lines.append(f"- **Source File**: `{metadata['filename']}`")

    if "sample_rate" in metadata:
        sr = metadata["sample_rate"]
        if sr >= 1e9:
            sr_str = f"{sr / 1e9:.3f} GS/s"
        elif sr >= 1e6:
            sr_str = f"{sr / 1e6:.3f} MS/s"
        elif sr >= 1e3:
            sr_str = f"{sr / 1e3:.3f} kS/s"
        else:
            sr_str = f"{sr:.3f} S/s"
        lines.append(f"- **Sample Rate**: {sr_str}")

    if "samples" in metadata:
        lines.append(f"- **Samples**: {metadata['samples']:,}")

    if "duration" in metadata:
        dur = metadata["duration"]
        if dur >= 1.0:
            dur_str = f"{dur:.3f} s"
        elif dur >= 1e-3:
            dur_str = f"{dur * 1e3:.3f} ms"
        elif dur >= 1e-6:
            dur_str = f"{dur * 1e6:.3f} us"
        else:
            dur_str = f"{dur * 1e9:.3f} ns"
        lines.append(f"- **Duration**: {dur_str}")

    lines.append("")
    return lines


def _generate_measurements_section(measurements: dict[str, Any]) -> list[str]:
    """Generate measurements table section."""
    lines = ["## Measurement Results\n", ""]

    if not measurements:
        lines.append("*No measurements available.*\n")
        return lines

    # Create table header
    lines.append("| Parameter | Value | Unit | Status |")
    lines.append("|-----------|-------|------|--------|")

    for name, value in measurements.items():
        if isinstance(value, dict):
            # Structured measurement with value, unit, status
            val = value.get("value", "N/A")
            unit = value.get("unit", "")
            status = value.get("status", "")

            # Format value
            val_str = _format_value(val, unit) if isinstance(val, float) else str(val)

            # Format status with emoji
            if status.upper() == "PASS":
                status_str = "PASS"
            elif status.upper() == "FAIL":
                status_str = "FAIL"
            elif status.upper() == "WARNING":
                status_str = "WARNING"
            else:
                status_str = status

            lines.append(f"| {name} | {val_str} | {unit} | {status_str} |")
        else:
            # Simple value
            val_str = f"{value:.6g}" if isinstance(value, float) else str(value)
            lines.append(f"| {name} | {val_str} | - | - |")

    lines.append("")
    return lines


def _format_value(value: float, unit: str) -> str:
    """Format value with appropriate SI prefix."""
    if value == 0:
        return "0"

    abs_val = abs(value)

    # Time units
    if unit in ("s", "sec", "seconds"):
        if abs_val >= 1.0:
            return f"{value:.6g}"
        elif abs_val >= 1e-3:
            return f"{value * 1e3:.6g} m"
        elif abs_val >= 1e-6:
            return f"{value * 1e6:.6g} u"
        elif abs_val >= 1e-9:
            return f"{value * 1e9:.6g} n"
        else:
            return f"{value * 1e12:.6g} p"

    # Frequency units
    if unit in ("Hz", "hz"):
        if abs_val >= 1e9:
            return f"{value / 1e9:.6g} G"
        elif abs_val >= 1e6:
            return f"{value / 1e6:.6g} M"
        elif abs_val >= 1e3:
            return f"{value / 1e3:.6g} k"
        else:
            return f"{value:.6g}"

    # Voltage units
    if unit in ("V", "v", "volts"):
        if abs_val >= 1.0:
            return f"{value:.6g}"
        elif abs_val >= 1e-3:
            return f"{value * 1e3:.6g} m"
        elif abs_val >= 1e-6:
            return f"{value * 1e6:.6g} u"
        else:
            return f"{value * 1e9:.6g} n"

    # Default formatting
    return f"{value:.6g}"


def _generate_plots_section(
    plots: list[Any],
    report_path: str | Path,
    embed_images: bool,
) -> list[str]:
    """Generate plots section."""
    lines = ["## Plots and Visualizations\n", ""]

    report_path = Path(report_path)
    plots_dir = report_path.parent / f"{report_path.stem}_plots"

    for i, plot in enumerate(plots, start=1):
        if isinstance(plot, dict):
            # Plot with metadata
            fig = plot.get("figure")
            caption = plot.get("caption", f"Figure {i}")
            alt_text = plot.get("alt_text", caption)
        else:
            fig = plot
            caption = f"Figure {i}"
            alt_text = caption

        if fig is None:
            continue

        if isinstance(fig, str | Path):
            # Path to existing image
            if embed_images:
                # Read and embed as base64
                try:
                    img_data = Path(fig).read_bytes()
                    img_ext = Path(fig).suffix.lower()
                    mime_type = {
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".svg": "image/svg+xml",
                    }.get(img_ext, "image/png")

                    b64 = base64.b64encode(img_data).decode("utf-8")
                    lines.append(f"### {caption}\n")
                    lines.append(f"![{alt_text}](data:{mime_type};base64,{b64})\n")
                except Exception:
                    lines.append(f"### {caption}\n")
                    lines.append(f"*Unable to embed image: {fig}*\n")
            else:
                lines.append(f"### {caption}\n")
                lines.append(f"![{alt_text}]({fig})\n")
        else:
            # Matplotlib figure
            try:
                if embed_images:
                    # Embed as base64 PNG
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode("utf-8")
                    lines.append(f"### {caption}\n")
                    lines.append(f"![{alt_text}](data:image/png;base64,{b64})\n")
                else:
                    # Save to separate file
                    plots_dir.mkdir(exist_ok=True)
                    plot_path = plots_dir / f"figure_{i}.png"
                    fig.savefig(plot_path, format="png", dpi=150, bbox_inches="tight")
                    rel_path = plot_path.relative_to(report_path.parent)
                    lines.append(f"### {caption}\n")
                    lines.append(f"![{alt_text}]({rel_path})\n")
            except Exception as e:
                lines.append(f"### {caption}\n")
                lines.append(f"*Unable to render figure: {e}*\n")

        lines.append("")

    return lines


def generate_markdown_report(
    data: dict[str, Any],
    *,
    title: str = "Oscura Analysis Report",
    author: str | None = None,
    include_plots: bool = True,
    embed_images: bool = True,
    sections: list[str] | None = None,
) -> str:
    """Generate Markdown report as string.

    Args:
        data: Dictionary containing measurement results, plots, and metadata.
        title: Report title.
        author: Author name (optional).
        include_plots: Include plots in report.
        embed_images: Embed images as base64.
        sections: List of sections to include.

    Returns:
        Markdown content as string.

    References:
        EXP-006
    """
    lines: list[str] = []

    # Header
    lines.append(f"# {title}\n")
    lines.append("")

    # Metadata section
    if sections is None or "metadata" in sections:
        lines.extend(_generate_metadata_section(data, author))

    # Executive summary
    if (sections is None or "summary" in sections) and "summary" in data:
        lines.append("## Executive Summary\n")
        lines.append(data["summary"])
        lines.append("")

    # Measurements table
    if (sections is None or "measurements" in sections) and "measurements" in data:
        lines.extend(_generate_measurements_section(data["measurements"]))

    # For string generation, only include plots if embed_images is True
    if include_plots and embed_images and (sections is None or "plots" in sections):
        if "plots" in data:
            # Simplified plot handling for string output
            lines.append("## Plots and Visualizations\n")
            lines.append("")
            for i, plot in enumerate(data["plots"], start=1):
                if isinstance(plot, dict):
                    caption = plot.get("caption", f"Figure {i}")
                else:
                    caption = f"Figure {i}"
                lines.append(f"### {caption}\n")
                lines.append("*[Embedded plot - save to file to view]*\n")
                lines.append("")

    # Conclusions
    if (sections is None or "conclusions" in sections) and "conclusions" in data:
        lines.append("## Conclusions\n")
        lines.append(data["conclusions"])
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "export_markdown",
    "generate_markdown_report",
]
