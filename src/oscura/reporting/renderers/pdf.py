"""Enhanced PDF report generation.

High-quality PDF report generation with embedded plots, metadata,
and PDF/A compliance for archival (enhancements to existing PDF module).


References:
    - REPORT-008: PDF Report Generation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oscura.reporting.core import Report


@dataclass
class PDFRenderer:
    """PDF report renderer.

    Attributes:
        dpi: Plot rendering DPI (default 300 for print quality).
        embed_fonts: Embed fonts for consistency (default True).
        vector_graphics: Use vector graphics for plots (default True).
        table_of_contents: Include TOC (default True).
        pdfa_compliance: Generate PDF/A-1b output (default False).
        page_numbering: Include page numbers (default True).

    References:
        REPORT-008: PDF Report Generation Engine
    """

    dpi: int = 300
    embed_fonts: bool = True
    vector_graphics: bool = True
    table_of_contents: bool = True
    pdfa_compliance: bool = False
    page_numbering: bool = True


def render_to_pdf(
    report: Report,
    output_path: str | None = None,
    **kwargs: Any,
) -> bytes:
    """Render report to PDF.

    Args:
        report: Report object to render.
        output_path: Optional output path (if None, returns bytes).
        **kwargs: Additional PDF rendering options.

    Returns:
        PDF data as bytes (if output_path is None).

    Example:
        >>> from oscura.reporting.core import Report, ReportConfig
        >>> report = Report(config=ReportConfig(title="Test Report"))
        >>> pdf_bytes = render_to_pdf(report)

    References:
        REPORT-008: PDF Report Generation Engine
    """
    # Import existing PDF module
    from oscura.reporting.pdf import generate_pdf_report

    # Merge kwargs with defaults
    renderer = PDFRenderer(**kwargs)

    # Generate PDF
    pdf_bytes = generate_pdf_report(
        report,
        dpi=renderer.dpi,
        embed_fonts=renderer.embed_fonts,
        vector_graphics=renderer.vector_graphics,
        table_of_contents=renderer.table_of_contents,
        pdfa_compliance=renderer.pdfa_compliance,
    )

    # Save if path provided
    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes


__all__ = [
    "PDFRenderer",
    "render_to_pdf",
]
