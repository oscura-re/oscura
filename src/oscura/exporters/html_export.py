"""HTML report export for Oscura.

This module provides interactive HTML report generation with embedded Plotly charts,
measurement tables, and custom styling/theming.


Example:
    >>> from oscura.exporters.html_export import export_html
    >>> export_html(measurements, "report.html", title="Analysis Report")
"""

from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

# HTML template with modern styling
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="generator" content="Oscura Export">
    <title>{title}</title>
    {plotly_script}
    <style>
        :root {{
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --bg-color: #ffffff;
            --text-color: #333333;
            --border-color: #dddddd;
            --table-header-bg: #f2f2f2;
            --table-alt-row-bg: #f9f9f9;
        }}

        {dark_mode_styles}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            font-size: 14px;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--bg-color);
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            margin-bottom: 30px;
            border-bottom: 3px solid var(--primary-color);
            padding-bottom: 20px;
        }}

        h1 {{
            font-size: 28px;
            color: var(--primary-color);
            margin-bottom: 10px;
        }}

        h2 {{
            font-size: 20px;
            color: var(--primary-color);
            margin-top: 30px;
            margin-bottom: 15px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
        }}

        h3 {{
            font-size: 16px;
            color: var(--primary-color);
            margin-top: 20px;
            margin-bottom: 10px;
        }}

        .metadata {{
            background-color: var(--table-alt-row-bg);
            padding: 15px;
            border-radius: 5px;
            font-size: 13px;
            color: #666;
        }}

        .metadata span {{
            margin-right: 20px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}

        th, td {{
            padding: 10px 12px;
            text-align: left;
            border: 1px solid var(--border-color);
        }}

        th {{
            background-color: var(--table-header-bg);
            font-weight: 600;
        }}

        tr:nth-child(even) {{
            background-color: var(--table-alt-row-bg);
        }}

        tr:hover {{
            background-color: rgba(52, 152, 219, 0.1);
        }}

        .pass {{
            color: var(--success-color);
            font-weight: 600;
        }}

        .pass::before {{
            content: '\\2713 ';
        }}

        .fail {{
            color: var(--danger-color);
            font-weight: 600;
        }}

        .fail::before {{
            content: '\\2717 ';
        }}

        .warning {{
            color: var(--warning-color);
            font-weight: 600;
        }}

        .summary {{
            background-color: rgba(52, 152, 219, 0.1);
            border-left: 4px solid var(--secondary-color);
            padding: 15px;
            margin: 20px 0;
        }}

        .plot-container {{
            margin: 20px 0;
            padding: 15px;
            background-color: var(--bg-color);
            border: 1px solid var(--border-color);
            border-radius: 5px;
        }}

        .plot-container img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}

        .plot-caption {{
            text-align: center;
            font-style: italic;
            margin-top: 10px;
            color: #666;
        }}

        footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            text-align: center;
            font-size: 12px;
            color: #888;
        }}

        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}

            h1 {{
                font-size: 22px;
            }}

            table {{
                font-size: 12px;
            }}

            th, td {{
                padding: 6px 8px;
            }}
        }}

        @media print {{
            body {{
                padding: 0;
            }}

            .container {{
                max-width: 100%;
            }}
        }}
    </style>
</head>
<body{body_class}>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <div class="metadata">
                {metadata_html}
            </div>
        </header>

        {summary_html}

        {measurements_html}

        {plots_html}

        {conclusions_html}

        <footer>
            Generated by Oscura &middot; {timestamp}
        </footer>
    </div>
</body>
</html>
"""
DARK_MODE_CSS = """
        @media (prefers-color-scheme: dark) {
            :root {
                --bg-color: #1e1e1e;
                --text-color: #e0e0e0;
                --border-color: #444444;
                --table-header-bg: #2d2d2d;
                --table-alt-row-bg: #252525;
            }
        }

        body.dark-mode {
            --bg-color: #1e1e1e;
            --text-color: #e0e0e0;
            --border-color: #444444;
            --table-header-bg: #2d2d2d;
            --table-alt-row-bg: #252525;
        }
"""


def export_html(
    data: dict[str, Any],
    path: str | Path,
    *,
    title: str = "Oscura Analysis Report",
    author: str | None = None,
    include_plots: bool = True,
    self_contained: bool = True,
    interactive: bool = True,
    dark_mode: bool = False,
    theme: str | None = None,
) -> None:
    """Export measurement results to interactive HTML format.

    Args:
        data: Dictionary containing measurement results, plots, and metadata.
            Expected keys:
            - "measurements": dict of name -> value pairs
            - "plots": list of matplotlib/plotly figures or paths
            - "metadata": optional dict of metadata
            - "summary": optional executive summary text
            - "conclusions": optional conclusions text
        path: Output file path.
        title: Report title.
        author: Author name (optional).
        include_plots: Include plots in report.
        self_contained: Embed all resources inline (True) or save separately.
        interactive: Use Plotly for interactive charts when available.
        dark_mode: Enable dark mode styling.
        theme: Custom theme name (reserved for future use).

    References:
        EXP-007
    """
    html_content = generate_html_report(
        data,
        title=title,
        author=author,
        include_plots=include_plots,
        self_contained=self_contained,
        interactive=interactive,
        dark_mode=dark_mode,
        theme=theme,
    )

    Path(path).write_text(html_content, encoding="utf-8")


def generate_html_report(
    data: dict[str, Any],
    *,
    title: str = "Oscura Analysis Report",
    author: str | None = None,
    include_plots: bool = True,
    self_contained: bool = True,
    interactive: bool = True,
    dark_mode: bool = False,
    theme: str | None = None,
) -> str:
    """Generate HTML report as string.

    Args:
        data: Dictionary containing measurement results, plots, and metadata.
        title: Report title.
        author: Author name (optional).
        include_plots: Include plots in report.
        self_contained: Embed all resources inline.
        interactive: Use Plotly for interactive charts when available.
        dark_mode: Enable dark mode styling.
        theme: Custom theme name (reserved for future use).

    Returns:
        HTML content as string.

    References:
        EXP-007
    """
    # Metadata HTML
    metadata_parts = []
    metadata = data.get("metadata", {})

    if author:
        metadata_parts.append(f"<span><strong>Author:</strong> {author}</span>")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata_parts.append(f"<span><strong>Generated:</strong> {timestamp}</span>")

    if "filename" in metadata:
        metadata_parts.append(
            f"<span><strong>Source:</strong> {_html_escape(metadata['filename'])}</span>"
        )

    if "sample_rate" in metadata:
        sr = metadata["sample_rate"]
        sr_str = _format_sample_rate(sr)
        metadata_parts.append(f"<span><strong>Sample Rate:</strong> {sr_str}</span>")

    if "samples" in metadata:
        metadata_parts.append(f"<span><strong>Samples:</strong> {metadata['samples']:,}</span>")

    metadata_html = "\n                ".join(metadata_parts)

    # Summary HTML
    summary_html = ""
    if "summary" in data:
        summary_html = f"""
        <section id="summary">
            <h2>Executive Summary</h2>
            <div class="summary">
                <p>{_html_escape(data["summary"])}</p>
            </div>
        </section>
        """
    # Measurements HTML
    measurements_html = ""
    if "measurements" in data:
        measurements_html = _generate_measurements_html(data["measurements"])

    # Plots HTML
    plots_html = ""
    plotly_script = ""
    if include_plots and "plots" in data:
        plots_html, plotly_script = _generate_plots_html(data["plots"], self_contained, interactive)

    # Conclusions HTML
    conclusions_html = ""
    if "conclusions" in data:
        conclusions_html = f"""
        <section id="conclusions">
            <h2>Conclusions</h2>
            <p>{_html_escape(data["conclusions"])}</p>
        </section>
        """
    # Dark mode styles
    dark_mode_styles = DARK_MODE_CSS if dark_mode else ""

    # Body class for dark mode
    body_class = ' class="dark-mode"' if dark_mode else ""

    # Generate final HTML
    html = HTML_TEMPLATE.format(
        title=_html_escape(title),
        plotly_script=plotly_script,
        dark_mode_styles=dark_mode_styles,
        body_class=body_class,
        metadata_html=metadata_html,
        summary_html=summary_html,
        measurements_html=measurements_html,
        plots_html=plots_html,
        conclusions_html=conclusions_html,
        timestamp=timestamp,
    )

    return html


def _html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _format_sample_rate(sr: float) -> str:
    """Format sample rate with SI prefix."""
    if sr >= 1e9:
        return f"{sr / 1e9:.3f} GS/s"
    elif sr >= 1e6:
        return f"{sr / 1e6:.3f} MS/s"
    elif sr >= 1e3:
        return f"{sr / 1e3:.3f} kS/s"
    else:
        return f"{sr:.3f} S/s"


def _format_value(value: float, unit: str) -> str:
    """Format value with appropriate precision."""
    if value == 0:
        return "0"

    abs_val = abs(value)

    # Time units
    if unit in ("s", "sec", "seconds"):
        if abs_val >= 1.0:
            return f"{value:.6g} s"
        elif abs_val >= 1e-3:
            return f"{value * 1e3:.6g} ms"
        elif abs_val >= 1e-6:
            return f"{value * 1e6:.6g} us"
        elif abs_val >= 1e-9:
            return f"{value * 1e9:.6g} ns"
        else:
            return f"{value * 1e12:.6g} ps"

    # Frequency units
    if unit in ("Hz", "hz"):
        if abs_val >= 1e9:
            return f"{value / 1e9:.6g} GHz"
        elif abs_val >= 1e6:
            return f"{value / 1e6:.6g} MHz"
        elif abs_val >= 1e3:
            return f"{value / 1e3:.6g} kHz"
        else:
            return f"{value:.6g} Hz"

    # Default formatting
    if unit:
        return f"{value:.6g} {unit}"
    return f"{value:.6g}"


def _generate_measurements_html(measurements: dict[str, Any]) -> str:
    """Generate measurements table HTML."""
    if not measurements:
        return ""

    rows = []
    for name, value in measurements.items():
        if isinstance(value, dict):
            val = value.get("value", "N/A")
            unit = value.get("unit", "")
            status = value.get("status", "")

            val_str = _format_value(val, unit) if isinstance(val, float) else str(val)

            # Status class and formatting
            status_upper = str(status).upper()
            if status_upper == "PASS":
                status_html = '<span class="pass">PASS</span>'
            elif status_upper == "FAIL":
                status_html = '<span class="fail">FAIL</span>'
            elif status_upper == "WARNING":
                status_html = '<span class="warning">WARNING</span>'
            else:
                status_html = _html_escape(str(status))

            rows.append(
                f"<tr><td>{_html_escape(name)}</td>"
                f"<td>{_html_escape(val_str)}</td>"
                f"<td>{_html_escape(unit)}</td>"
                f"<td>{status_html}</td></tr>"
            )
        else:
            val_str = f"{value:.6g}" if isinstance(value, float) else str(value)

            rows.append(
                f"<tr><td>{_html_escape(name)}</td>"
                f"<td>{_html_escape(val_str)}</td>"
                f"<td>-</td><td>-</td></tr>"
            )

    return f"""
        <section id="measurements">
            <h2>Measurement Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                        <th>Unit</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </section>
    """


def _generate_plots_html(
    plots: list[Any],
    self_contained: bool,
    interactive: bool,
) -> tuple[str, str]:
    """Generate plots HTML and Plotly script if needed.

    Args:
        plots: List of plot objects (matplotlib figures, plotly figures, or paths).
        self_contained: Embed all resources inline (True) or reference externally.
        interactive: Use Plotly for interactive charts when available.

    Returns:
        Tuple of (plots_html, plotly_script_tag)
    """
    if not plots:
        return "", ""

    plot_divs = []
    has_plotly = False

    for i, plot in enumerate(plots, start=1):
        if isinstance(plot, dict):
            fig = plot.get("figure")
            caption = plot.get("caption", f"Figure {i}")
        else:
            fig = plot
            caption = f"Figure {i}"

        if fig is None:
            continue

        # Check if it's a Plotly figure
        plotly_html = _try_render_plotly(fig, interactive)
        if plotly_html:
            has_plotly = True
            plot_divs.append(
                f'<div class="plot-container"><h3>{_html_escape(caption)}</h3>{plotly_html}</div>'
            )
            continue

        # Try matplotlib figure
        img_html = _try_render_matplotlib(fig, self_contained)
        if img_html:
            plot_divs.append(
                f'<div class="plot-container">'
                f"<h3>{_html_escape(caption)}</h3>"
                f"{img_html}"
                f'<div class="plot-caption">{_html_escape(caption)}</div>'
                f"</div>"
            )
            continue

        # Image path
        if isinstance(fig, str | Path):
            if self_contained:
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
                    img_html = (
                        f'<img src="data:{mime_type};base64,{b64}" alt="{_html_escape(caption)}">'
                    )
                except Exception:
                    img_html = f"<p><em>Unable to embed image: {fig}</em></p>"
            else:
                img_html = f'<img src="{_html_escape(str(fig))}" alt="{_html_escape(caption)}">'

            plot_divs.append(
                f'<div class="plot-container">'
                f"<h3>{_html_escape(caption)}</h3>"
                f"{img_html}"
                f'<div class="plot-caption">{_html_escape(caption)}</div>'
                f"</div>"
            )

    # Plotly CDN script (only included if we have Plotly figures)
    plotly_script = ""
    if has_plotly:
        plotly_script = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

    plots_html = (
        f"""
        <section id="plots">
            <h2>Plots and Visualizations</h2>
            {"".join(plot_divs)}
        </section>
    """
        if plot_divs
        else ""
    )

    return plots_html, plotly_script


def _try_render_plotly(fig: Any, interactive: bool) -> str | None:
    """Try to render a Plotly figure to HTML.

    Args:
        fig: Figure object to render (may be Plotly figure or other type).
        interactive: Enable interactive Plotly rendering.

    Returns:
        HTML string if successful, None if not a Plotly figure.
    """
    if not interactive:
        return None

    try:
        import plotly.graph_objects as go  # type: ignore[import-not-found]

        if isinstance(fig, go.Figure):
            return fig.to_html(  # type: ignore[no-any-return]
                full_html=False,
                include_plotlyjs=False,
                config={"displayModeBar": True, "responsive": True},
            )
    except ImportError:
        pass

    return None


def _try_render_matplotlib(fig: Any, self_contained: bool) -> str | None:
    """Try to render a Matplotlib figure to HTML.

    Args:
        fig: Figure object to render (may be matplotlib figure or other type).
        self_contained: Embed image as base64 data URI.

    Returns:
        HTML img tag if successful, None if not a Matplotlib figure.
    """
    try:
        import matplotlib.pyplot as plt

        if hasattr(fig, "savefig"):
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("utf-8")
            return f'<img src="data:image/png;base64,{b64}" alt="Figure">'
    except ImportError:
        pass
    except Exception:
        pass

    return None


__all__ = [
    "export_html",
    "generate_html_report",
]
