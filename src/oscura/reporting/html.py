"""HTML report generation for Oscura.

This module provides professional HTML report generation with modern features
including responsive design, interactive plots, and collapsible sections.

Features:
    - Professional formatting standards
    - Visual emphasis system
    - Smart content filtering (interactive)
    - Modern HTML with progressive disclosure
    - Collapsible sections mechanism

Example:
    >>> from oscura.reporting.html import generate_html_report
    >>> html = generate_html_report(report, interactive=True, dark_mode=False)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oscura.reporting.core import Report


def generate_html_report(
    report: Report,
    *,
    interactive: bool = True,
    dark_mode: bool = False,
    collapsible_sections: bool = True,
    responsive: bool = True,
    self_contained: bool = True,
) -> str:
    """Generate modern HTML report.

    Args:
        report: Report object to render.
        interactive: Enable interactive features (sorting, filtering).
        dark_mode: Include dark mode support.
        collapsible_sections: Make sections collapsible.
        responsive: Enable responsive design for mobile.
        self_contained: Include all assets inline (no external dependencies).

    Returns:
        HTML string.
    """
    html_parts = [
        _generate_html_header(report, dark_mode, responsive),
        _generate_html_styles(dark_mode, responsive),
        _generate_html_scripts() if interactive or collapsible_sections else "",
        "</head>",
        "<body>",
        _generate_html_nav(report) if len(report.sections) > 3 else "",
        '<div class="container">',
        f"<header><h1>{report.config.title}</h1>",
        _generate_metadata_section(report),
        "</header>",
        _generate_html_content(report, collapsible_sections),
        "</div>",
        "</body>",
        "</html>",
    ]

    return "\n".join(html_parts)


def _generate_html_header(report: Report, dark_mode: bool, responsive: bool) -> str:
    """Generate HTML header."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="author" content="{report.config.author or "Oscura"}">
    <meta name="generator" content="Oscura Reporting System">
    <title>{report.config.title}</title>"""


def _generate_html_styles(dark_mode: bool, responsive: bool) -> str:
    """Generate CSS styles for HTML report."""
    styles = """
<style>
/* Professional Formatting Standards */
:root {
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
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    body.dark-mode {
        --bg-color: #1e1e1e;
        --text-color: #e0e0e0;
        --border-color: #444444;
        --table-header-bg: #2d2d2d;
        --table-alt-row-bg: #252525;
    }
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 10pt;
    line-height: 1.5;
    color: var(--text-color);
    background-color: var(--bg-color);
    margin: 0;
    padding: 0;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1in;
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: Arial, Helvetica, sans-serif;
    line-height: 1.2;
    margin-top: 1em;
    margin-bottom: 0.5em;
    color: var(--primary-color);
}

h1 { font-size: 24pt; }
h2 { font-size: 18pt; border-bottom: 2px solid var(--border-color); padding-bottom: 0.3em; }
h3 { font-size: 14pt; }
h4 { font-size: 12pt; }

code, pre {
    font-family: 'Courier New', Courier, monospace;
    background-color: var(--table-alt-row-bg);
    padding: 2px 4px;
    border-radius: 3px;
}

pre {
    padding: 10px;
    overflow-x: auto;
}

/* Visual Emphasis */
.pass {
    color: var(--success-color);
    font-weight: bold;
}

.fail {
    color: var(--danger-color);
    font-weight: bold;
}

.warning {
    color: var(--warning-color);
    font-weight: bold;
}

.pass::before { content: '\\2713 '; }
.fail::before { content: '\\2717 '; }

/* Severity indicators */
.severity-critical {
    background-color: rgba(231, 76, 60, 0.2);
    border-left: 4px solid var(--danger-color);
    padding: 10px;
    margin: 10px 0;
}

.severity-warning {
    background-color: rgba(243, 156, 18, 0.2);
    border-left: 4px solid var(--warning-color);
    padding: 10px;
    margin: 10px 0;
}

.severity-info {
    background-color: rgba(52, 152, 219, 0.2);
    border-left: 4px solid var(--secondary-color);
    padding: 10px;
    margin: 10px 0;
}

/* Callout boxes */
.callout {
    background-color: rgba(241, 196, 15, 0.15);
    border: 1px solid var(--warning-color);
    border-radius: 5px;
    padding: 15px;
    margin: 15px 0;
}

.callout-title {
    font-weight: bold;
    margin-bottom: 10px;
}

/* Tables */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
    font-size: 10pt;
}

th, td {
    border: 1px solid var(--border-color);
    padding: 8px 12px;
    text-align: left;
}

th {
    background-color: var(--table-header-bg);
    font-weight: bold;
    font-family: Arial, Helvetica, sans-serif;
}

tr:nth-child(even) {
    background-color: var(--table-alt-row-bg);
}

tr:hover {
    background-color: rgba(52, 152, 219, 0.1);
}

caption {
    caption-side: bottom;
    font-style: italic;
    padding: 8px;
    text-align: left;
}

/* Collapsible sections */
.collapsible {
    cursor: pointer;
    user-select: none;
    display: flex;
    align-items: center;
    gap: 8px;
}

.collapsible::before {
    content: '\\25BC';
    display: inline-block;
    transition: transform 0.3s;
}

.collapsible.collapsed::before {
    transform: rotate(-90deg);
}

.collapsible-content {
    max-height: 5000px;
    overflow: hidden;
    transition: max-height 0.3s ease-out;
}

.collapsible-content.collapsed {
    max-height: 0;
}

/* Metadata section */
.metadata {
    background-color: var(--table-alt-row-bg);
    padding: 15px;
    margin: 20px 0;
    border-radius: 5px;
    font-size: 9pt;
}

.metadata-item {
    display: inline-block;
    margin-right: 20px;
}

/* Navigation */
nav {
    background-color: var(--primary-color);
    color: white;
    padding: 15px;
    position: sticky;
    top: 0;
    z-index: 1000;
}

nav ul {
    list-style: none;
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}

nav a {
    color: white;
    text-decoration: none;
}

nav a:hover {
    text-decoration: underline;
}

/* Responsive design */
@media (max-width: 768px) {
    .container {
        padding: 0.5in;
    }

    h1 { font-size: 20pt; }
    h2 { font-size: 16pt; }
    h3 { font-size: 12pt; }

    table {
        font-size: 9pt;
    }

    nav ul {
        flex-direction: column;
        gap: 10px;
    }
}

/* Print styles */
@media print {
    .container {
        max-width: 100%;
        padding: 0;
    }

    nav {
        display: none;
    }

    .collapsible-content {
        max-height: none !important;
    }
}
</style>"""
    return styles


def _generate_html_scripts() -> str:
    """Generate JavaScript for interactive features."""
    return """
<script>
// Collapsible sections
document.addEventListener('DOMContentLoaded', function() {
    const collapsibles = document.querySelectorAll('.collapsible');

    collapsibles.forEach(function(collapsible) {
        collapsible.addEventListener('click', function() {
            this.classList.toggle('collapsed');
            const content = this.nextElementSibling;
            if (content && content.classList.contains('collapsible-content')) {
                content.classList.toggle('collapsed');
            }
        });
    });

    // Dark mode toggle
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', function() {
            document.body.classList.toggle('dark-mode');
        });
    }

    // Table sorting (if interactive)
    const tables = document.querySelectorAll('table.sortable');
    tables.forEach(function(table) {
        const headers = table.querySelectorAll('th');
        headers.forEach(function(header, index) {
            header.addEventListener('click', function() {
                sortTable(table, index);
            });
            header.style.cursor = 'pointer';
        });
    });
});

function sortTable(table, column) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));

    rows.sort(function(a, b) {
        const aText = a.cells[column].textContent.trim();
        const bText = b.cells[column].textContent.trim();

        // Try numeric comparison first
        const aNum = parseFloat(aText);
        const bNum = parseFloat(bText);

        if (!isNaN(aNum) && !isNaN(bNum)) {
            return aNum - bNum;
        }

        // Fall back to string comparison
        return aText.localeCompare(bText);
    });

    rows.forEach(function(row) {
        tbody.appendChild(row);
    });
}
</script>"""


def _generate_html_nav(report: Report) -> str:
    """Generate navigation menu."""
    nav_items = []
    for section in report.sections:
        if section.visible:
            section_id = section.title.lower().replace(" ", "-")
            nav_items.append(f'<li><a href="#{section_id}">{section.title}</a></li>')

    return f"""
<nav>
    <ul>
        {"".join(nav_items)}
    </ul>
</nav>"""


def _generate_metadata_section(report: Report) -> str:
    """Generate metadata section."""
    items = []
    if report.config.author:
        items.append(
            f'<span class="metadata-item"><strong>Author:</strong> {report.config.author}</span>'
        )
    items.append(
        f'<span class="metadata-item"><strong>Date:</strong> {report.config.created.strftime("%Y-%m-%d %H:%M")}</span>'
    )
    if report.config.verbosity:
        items.append(
            f'<span class="metadata-item"><strong>Detail Level:</strong> {report.config.verbosity}</span>'
        )

    return f'<div class="metadata">{" ".join(items)}</div>'


def _generate_html_content(report: Report, collapsible: bool) -> str:
    """Generate main content sections."""
    content = []

    for section in report.sections:
        if not section.visible:
            continue

        section_id = section.title.lower().replace(" ", "-")
        content.append(f'<section id="{section_id}">')

        # Section header
        tag = f"h{min(section.level + 1, 6)}"
        if collapsible and section.collapsible:
            content.append(f'<{tag} class="collapsible">{section.title}</{tag}>')
            content.append('<div class="collapsible-content">')
        else:
            content.append(f"<{tag}>{section.title}</{tag}>")

        # Section content
        if isinstance(section.content, str):
            content.append(f"<p>{section.content}</p>")
        elif isinstance(section.content, list):
            for item in section.content:
                if isinstance(item, dict):
                    if item.get("type") == "table":
                        content.append(_table_to_html(item))
                    elif item.get("type") == "figure":
                        content.append(_figure_to_html(item))
                else:
                    content.append(f"<p>{item}</p>")

        # Subsections
        for subsec in section.subsections:
            if not subsec.visible:
                continue
            sub_tag = f"h{min(subsec.level + 1, 6)}"
            content.append(f"<{sub_tag}>{subsec.title}</{sub_tag}>")
            if isinstance(subsec.content, str):
                content.append(f"<p>{subsec.content}</p>")

        if collapsible and section.collapsible:
            content.append("</div>")

        content.append("</section>")

    return "\n".join(content)


def _table_to_html(table: dict[str, Any]) -> str:
    """Convert table dictionary to HTML."""
    lines = ['<table class="sortable">']

    headers = table.get("headers", [])
    data = table.get("data", [])

    if headers:
        lines.append("<thead><tr>")
        for h in headers:
            lines.append(f"<th>{h}</th>")
        lines.append("</tr></thead>")

    lines.append("<tbody>")
    for row in data:
        lines.append("<tr>")
        for cell in row:
            # Apply visual emphasis for PASS/FAIL
            cell_str = str(cell)
            if "PASS" in cell_str.upper():
                lines.append(f'<td class="pass">{cell}</td>')
            elif "FAIL" in cell_str.upper():
                lines.append(f'<td class="fail">{cell}</td>')
            elif "WARNING" in cell_str.upper():
                lines.append(f'<td class="warning">{cell}</td>')
            else:
                lines.append(f"<td>{cell}</td>")
        lines.append("</tr>")
    lines.append("</tbody>")
    lines.append("</table>")

    if table.get("caption"):
        lines.append(f"<caption>{table['caption']}</caption>")

    return "\n".join(lines)


def _figure_to_html(figure: dict[str, Any]) -> str:
    """Convert figure dictionary to HTML."""
    width = figure.get("width", "100%")
    caption = figure.get("caption", "")

    html = f'<figure style="max-width: {width}; margin: 20px auto;">'

    # Handle different figure types
    fig_obj = figure.get("figure")
    if isinstance(fig_obj, str):
        # Assume it's a path to an image
        html += f'<img src="{fig_obj}" alt="{caption}" style="width: 100%;">'
    else:
        # Placeholder for matplotlib figures
        html += f'<div class="figure-placeholder">[Figure: {caption}]</div>'

    if caption:
        html += f"<figcaption>{caption}</figcaption>"

    html += "</figure>"
    return html


def save_html_report(
    report: Report,
    path: str | Path,
    **kwargs: Any,
) -> None:
    """Save report as HTML file.

    Args:
        report: Report object.
        path: Output file path.
        **kwargs: Additional options for generate_html_report.
    """
    html_content = generate_html_report(report, **kwargs)
    Path(path).write_text(html_content, encoding="utf-8")
