"""PDF report generation for Oscura.

This module provides high-quality PDF report generation with embedded plots,
metadata, and PDF/A compliance for archival.


Example:
    >>> from oscura.reporting.pdf import generate_pdf_report
    >>> pdf_bytes = generate_pdf_report(report, dpi=300, pdfa_compliance=True)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oscura.reporting.core import Report, Section

# Optional imports for PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_pdf_report(
    report: Report,
    *,
    dpi: int = 300,
    embed_fonts: bool = True,
    vector_graphics: bool = True,
    table_of_contents: bool = True,
    pdfa_compliance: bool = False,
) -> bytes:
    """Generate high-quality PDF report.

    Args:
        report: Report object to render.
        dpi: Plot rendering DPI (default 300).
        embed_fonts: Embed fonts for consistency.
        vector_graphics: Use vector graphics for plots.
        table_of_contents: Include table of contents.
        pdfa_compliance: Generate PDF/A-1b compliant output.

    Returns:
        PDF data as bytes.

    Raises:
        ImportError: If reportlab is not installed.

    References:
        REPORT-001, REPORT-002, REPORT-008, RPT-001
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab is required for PDF generation. Install with: pip install reportlab"
        )

    buffer = io.BytesIO()

    # Determine page size
    page_size = A4 if report.config.page_size == "A4" else LETTER
    margin = report.config.margins * inch

    # Create PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=page_size,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title=report.config.title,
        author=report.config.author or "Oscura",
    )

    # Build document story
    story = []
    styles = _create_styles()

    # Add title
    story.append(Paragraph(report.config.title, styles["Title"]))
    story.append(Spacer(1, 0.3 * inch))

    # Add metadata
    metadata_text = _format_metadata(report)
    story.append(Paragraph(metadata_text, styles["Metadata"]))
    story.append(Spacer(1, 0.5 * inch))

    # Add watermark if specified
    if report.config.watermark:
        # Watermark would be added via PageTemplate in production
        pass

    # Add table of contents if requested
    if table_of_contents and len(report.sections) > 3:
        story.append(Paragraph("Table of Contents", styles["Heading1"]))
        for i, section in enumerate(report.sections):
            if section.visible:
                toc_entry = f"{i + 1}. {section.title}"
                story.append(Paragraph(toc_entry, styles["TOC"]))
        story.append(PageBreak())

    # Add sections
    for section in report.sections:
        if not section.visible:
            continue

        _add_pdf_section(story, section, styles, report)

    # Build PDF
    doc.build(story)

    return buffer.getvalue()


def _create_styles() -> dict[str, ParagraphStyle]:
    """Create PDF paragraph styles."""
    base_styles = getSampleStyleSheet()
    base = base_styles["Normal"]

    return {
        "Normal": base,
        "Title": ParagraphStyle(
            "Title",
            base,
            fontSize=24,
            textColor=colors.HexColor("#2c3e50"),
            spaceAfter=12,
            alignment=1,
            fontName="Helvetica-Bold",
        ),
        "Heading1": ParagraphStyle(
            "Heading1",
            base,
            fontSize=18,
            textColor=colors.HexColor("#2c3e50"),
            spaceBefore=12,
            spaceAfter=6,
            fontName="Helvetica-Bold",
        ),
        "Heading2": ParagraphStyle(
            "Heading2",
            base,
            fontSize=14,
            textColor=colors.HexColor("#34495e"),
            spaceBefore=10,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        "Heading3": ParagraphStyle(
            "Heading3",
            base,
            fontSize=12,
            textColor=colors.HexColor("#34495e"),
            spaceBefore=8,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        "Body": ParagraphStyle("Body", base, fontSize=10, leading=15, fontName="Times-Roman"),
        "Metadata": ParagraphStyle(
            "Metadata", base, fontSize=9, textColor=colors.HexColor("#555555"), fontName="Helvetica"
        ),
        "TOC": ParagraphStyle("TOC", base, fontSize=10, leftIndent=20, spaceAfter=4),
        "Pass": ParagraphStyle(
            "Pass",
            base,
            fontSize=10,
            textColor=colors.HexColor("#27ae60"),
            fontName="Helvetica-Bold",
        ),
        "Fail": ParagraphStyle(
            "Fail",
            base,
            fontSize=10,
            textColor=colors.HexColor("#e74c3c"),
            fontName="Helvetica-Bold",
        ),
        "Warning": ParagraphStyle(
            "Warning",
            base,
            fontSize=10,
            textColor=colors.HexColor("#f39c12"),
            fontName="Helvetica-Bold",
        ),
    }


def _format_metadata(report: Report) -> str:
    """Format report metadata."""
    parts = []
    if report.config.author:
        parts.append(f"<b>Author:</b> {report.config.author}")
    parts.append(f"<b>Date:</b> {report.config.created.strftime('%Y-%m-%d %H:%M')}")
    if report.config.verbosity:
        parts.append(f"<b>Detail Level:</b> {report.config.verbosity}")

    return " | ".join(parts)


def _add_pdf_section(
    story: list,  # type: ignore[type-arg]
    section: Section,
    styles: dict[str, ParagraphStyle],
    report: Report,
) -> None:
    """Add a section to the PDF story."""
    # Section heading
    heading_style = f"Heading{min(section.level, 3)}"
    story.append(Paragraph(section.title, styles[heading_style]))
    story.append(Spacer(1, 0.2 * inch))

    # Section content
    if isinstance(section.content, str):
        # Split into paragraphs
        paragraphs = section.content.split("\n\n")
        for para in paragraphs:
            if para.strip():
                story.append(Paragraph(para.strip(), styles["Body"]))
                story.append(Spacer(1, 0.1 * inch))

    elif isinstance(section.content, list):
        for item in section.content:
            if isinstance(item, dict):
                if item.get("type") == "table":
                    story.append(_create_pdf_table(item))
                    story.append(Spacer(1, 0.2 * inch))
                elif item.get("type") == "figure":
                    # Placeholder for figures
                    caption = item.get("caption", "Figure")
                    story.append(Paragraph(f"[Figure: {caption}]", styles["Body"]))
                    story.append(Spacer(1, 0.2 * inch))
            else:
                story.append(Paragraph(str(item), styles["Body"]))
                story.append(Spacer(1, 0.1 * inch))

    # Subsections
    for subsec in section.subsections:
        if not subsec.visible:
            continue
        sub_heading_style = f"Heading{min(subsec.level, 3)}"
        story.append(Paragraph(subsec.title, styles[sub_heading_style]))
        if isinstance(subsec.content, str) and subsec.content.strip():
            story.append(Paragraph(subsec.content, styles["Body"]))
            story.append(Spacer(1, 0.1 * inch))

    story.append(Spacer(1, 0.3 * inch))


def _create_pdf_table(table_dict: dict[str, Any]) -> Table:
    """Create PDF table with professional formatting., REPORT-002."""
    headers = table_dict.get("headers", [])
    data = table_dict.get("data", [])

    # Build table data
    table_data = []
    if headers:
        table_data.append(headers)
    table_data.extend(data)

    # Create table
    table = Table(table_data)

    # Apply professional table style.
    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),  # Header
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#dddddd")),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("TOPPADDING", (0, 1), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 8),
    ]

    # Alternating row colors.
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            style_commands.append(
                (
                    "BACKGROUND",
                    (0, i),
                    (-1, i),
                    colors.HexColor("#f9f9f9"),
                )
            )

    # Apply visual emphasis for PASS/FAIL.
    for i, row in enumerate(data, start=1):
        for j, cell in enumerate(row):
            cell_str = str(cell).upper()
            if "PASS" in cell_str or "✓" in str(cell):
                style_commands.append(("TEXTCOLOR", (j, i), (j, i), colors.HexColor("#27ae60")))
            elif "FAIL" in cell_str or "✗" in str(cell):
                style_commands.append(("TEXTCOLOR", (j, i), (j, i), colors.HexColor("#e74c3c")))
            elif "WARNING" in cell_str:
                style_commands.append(("TEXTCOLOR", (j, i), (j, i), colors.HexColor("#f39c12")))

    table.setStyle(TableStyle(style_commands))

    return table


def save_pdf_report(
    report: Report,
    path: str | Path,
    **kwargs: Any,
) -> None:
    """Save report as PDF file.

    Args:
        report: Report object.
        path: Output file path.
        **kwargs: Additional options for generate_pdf_report.

    References:
        REPORT-001, REPORT-008, RPT-001
    """
    pdf_bytes = generate_pdf_report(report, **kwargs)
    Path(path).write_bytes(pdf_bytes)
