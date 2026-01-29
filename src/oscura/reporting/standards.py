"""Professional formatting standards and visual emphasis for reports.

This module provides professional formatting standards, visual emphasis
systems, and executive summary generation for Oscura reports.


Example:
    >>> from oscura.reporting.standards import FormatStandards, VisualEmphasis
    >>> standards = FormatStandards()
    >>> emphasis = VisualEmphasis()
    >>> formatted_text = emphasis.format_pass_fail(True)

References:
    REPORT-001, REPORT-002,
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class Severity(Enum):
    """Severity levels for findings and violations."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class ColorScheme(Enum):
    """Available colorblind-safe color schemes."""

    VIRIDIS = "viridis"
    CIVIDIS = "cividis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    OKABE_ITO = "okabe_ito"  # Colorblind-safe palette


@dataclass
class FormatStandards:
    """Professional formatting standards configuration.

    Defines typography, page layout, color schemes, and section hierarchy
    for professional report output.

    Attributes:
        heading_font: Font family for headings.
        body_font: Font family for body text.
        code_font: Font family for code and data.
        page_size: Page size (letter or A4).
        margins_inches: Page margins in inches.
        color_scheme: Colorblind-safe color palette.
        line_spacing: Line spacing multiplier.
        logo_max_height_inches: Maximum logo height.
        watermark_opacity: Watermark opacity (0.0-1.0).

    References:
        REPORT-001: Professional Formatting Standards
    """

    heading_font: str = "Arial, Helvetica, sans-serif"
    body_font: str = "Georgia, Times New Roman, serif"
    code_font: str = "Consolas, Courier New, monospace"
    page_size: Literal["letter", "A4"] = "letter"
    margins_inches: float = 1.0
    color_scheme: ColorScheme = ColorScheme.VIRIDIS
    line_spacing: float = 1.5
    logo_max_height_inches: float = 2.0
    watermark_opacity: float = 0.3

    # Font sizes in points
    title_size: int = 24
    h1_size: int = 18
    h2_size: int = 14
    h3_size: int = 12
    body_size: int = 10

    def to_css(self) -> str:
        """Generate CSS stylesheet from format standards.

        Returns:
            CSS stylesheet string.

        References:
            REPORT-001: Professional Formatting Standards
        """
        root_vars = self._generate_css_variables()
        base_styles = self._generate_css_base_styles()
        component_styles = self._generate_css_component_styles()
        print_styles = self._generate_css_print_styles()

        return f"""
/* Oscura Professional Report Styles - REPORT-001 */
{root_vars}
{base_styles}
{component_styles}
{print_styles}
"""

    def _generate_css_variables(self) -> str:
        """Generate CSS custom properties."""
        return f""":root {{
    --heading-font: {self.heading_font};
    --body-font: {self.body_font};
    --code-font: {self.code_font};
    --title-size: {self.title_size}pt;
    --h1-size: {self.h1_size}pt;
    --h2-size: {self.h2_size}pt;
    --h3-size: {self.h3_size}pt;
    --body-size: {self.body_size}pt;
    --line-spacing: {self.line_spacing};
    --margin: {self.margins_inches}in;

    /* Colorblind-safe palette */
    --color-pass: #2e7d32;
    --color-fail: #c62828;
    --color-warning: #f57c00;
    --color-info: #1565c0;
    --color-critical-bg: #ffebee;
    --color-warning-bg: #fff3e0;
    --color-info-bg: #e3f2fd;
}}"""

    def _generate_css_base_styles(self) -> str:
        """Generate base body and typography CSS."""
        return """
body {
    font-family: var(--body-font);
    font-size: var(--body-size);
    line-height: var(--line-spacing);
    margin: var(--margin);
    max-width: 8.5in;
    color: #333;
}

h1, h2, h3, h4 {
    font-family: var(--heading-font);
    line-height: 1.2;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
}

h1 { font-size: var(--h1-size); }
h2 { font-size: var(--h2-size); }
h3 { font-size: var(--h3-size); }

.report-title {
    font-size: var(--title-size);
    text-align: center;
    margin-bottom: 2em;
}

code, pre {
    font-family: var(--code-font);
    font-size: 0.9em;
    background-color: #f5f5f5;
    padding: 2px 4px;
    border-radius: 3px;
}"""

    def _generate_css_component_styles(self) -> str:
        """Generate component CSS (tables, severity, etc)."""
        tables = self._generate_css_tables()
        indicators = self._generate_css_indicators()
        callouts = self._generate_css_callouts()
        summary = self._generate_css_executive_summary()
        watermark = self._generate_css_watermark()

        return f"{tables}\n{indicators}\n{callouts}\n{summary}\n{watermark}"

    def _generate_css_tables(self) -> str:
        """Generate table CSS."""
        return """
/* Table styles */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
}

th, td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}

th {
    background-color: #f2f2f2;
    font-weight: bold;
}

tr:nth-child(even) {
    background-color: #f9f9f9;
}"""

    def _generate_css_indicators(self) -> str:
        """Generate pass/fail and severity indicator CSS."""
        return """
/* Pass/Fail indicators (REPORT-002) */
.pass {
    color: var(--color-pass);
}

.fail {
    color: var(--color-fail);
}

.warning {
    color: var(--color-warning);
}

/* Severity indicators (REPORT-002) */
.severity-critical {
    background-color: var(--color-critical-bg);
    border-left: 4px solid var(--color-fail);
    padding: 10px;
    margin: 10px 0;
}

.severity-warning {
    background-color: var(--color-warning-bg);
    border-left: 4px solid var(--color-warning);
    padding: 10px;
    margin: 10px 0;
}

.severity-info {
    background-color: var(--color-info-bg);
    border-left: 4px solid var(--color-info);
    padding: 10px;
    margin: 10px 0;
}"""

    def _generate_css_callouts(self) -> str:
        """Generate callout box CSS."""
        return """
/* Callout box (REPORT-002) */
.callout {
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 15px;
    margin: 15px 0;
    background-color: #fafafa;
}

.callout.key-finding {
    border-color: var(--color-info);
    background-color: var(--color-info-bg);
}

/* Highlighting for out-of-spec values */
.out-of-spec {
    background-color: rgba(255, 235, 59, 0.15);
    padding: 2px 4px;
    border-radius: 2px;
}"""

    def _generate_css_executive_summary(self) -> str:
        """Generate executive summary CSS."""
        return """
/* Executive summary styles (REPORT-004) */
.executive-summary {
    background-color: #f5f5f5;
    padding: 20px;
    margin: 20px 0;
    border-radius: 4px;
}

.executive-summary h2 {
    margin-top: 0;
}

.key-findings {
    list-style-type: none;
    padding-left: 0;
}

.key-findings li {
    padding: 5px 0;
    padding-left: 25px;
    position: relative;
}

.key-findings li::before {
    content: "";
    position: absolute;
    left: 0;
    top: 8px;
    width: 16px;
    height: 16px;
}

.key-findings li.critical::before {
    content: "!";
    color: var(--color-fail);
    font-weight: bold;
}"""

    def _generate_css_watermark(self) -> str:
        """Generate watermark CSS."""
        return f"""
/* Watermark */
.watermark {{
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) rotate(-45deg);
    font-size: 72pt;
    color: rgba(0, 0, 0, {self.watermark_opacity});
    pointer-events: none;
    z-index: 1000;
}}"""

    def _generate_css_print_styles(self) -> str:
        """Generate print media query CSS."""
        return """
/* Print styles */
@media print {
    body {
        margin: 0;
    }
    .page-break {
        page-break-before: always;
    }
    .no-print {
        display: none;
    }
}"""


@dataclass
class VisualEmphasis:
    """Visual emphasis system for pass/fail indicators and severity levels.

    Provides WCAG-compliant visual indicators using both symbols and colors
    for accessibility.

    Attributes:
        use_unicode_symbols: Use Unicode check/X marks.
        colorblind_safe: Always use symbols + colors (never color alone).
        highlight_violations: Highlight out-of-spec values.
        severity_icons: Show icons for severity levels.

    References:
        REPORT-002: Visual Emphasis System
    """

    use_unicode_symbols: bool = True
    colorblind_safe: bool = True
    highlight_violations: bool = True
    severity_icons: bool = True

    # Unicode symbols for status indicators
    CHECK_SYMBOL = "\u2713"  # Check mark
    CROSS_SYMBOL = "\u2717"  # X mark
    WARNING_SYMBOL = "\u26a0"  # Warning triangle
    INFO_SYMBOL = "\u2139"  # Info circle
    CRITICAL_SYMBOL = "\u2757"  # Exclamation mark

    def format_pass_fail(
        self,
        passed: bool,
        *,
        with_text: bool = True,
        html: bool = False,
    ) -> str:
        """Format pass/fail status with visual emphasis.

        Args:
            passed: Whether the test passed.
            with_text: Include PASS/FAIL text.
            html: Output as HTML with styling.

        Returns:
            Formatted status string.

        References:
            REPORT-002: Visual Emphasis System
        """
        if passed:
            symbol = self.CHECK_SYMBOL if self.use_unicode_symbols else "[PASS]"
            text = "PASS" if with_text else ""
            css_class = "pass"
        else:
            symbol = self.CROSS_SYMBOL if self.use_unicode_symbols else "[FAIL]"
            text = "FAIL" if with_text else ""
            css_class = "fail"

        result = f"{symbol} {text}".strip() if with_text else symbol

        if html:
            return f'<span class="{css_class}">{result}</span>'
        return result

    def format_severity(
        self,
        severity: Severity | str,
        message: str,
        *,
        html: bool = False,
    ) -> str:
        """Format message with severity indicator.

        Args:
            severity: Severity level.
            message: Message text.
            html: Output as HTML with styling.

        Returns:
            Formatted message string.

        References:
            REPORT-002: Visual Emphasis System
        """
        if isinstance(severity, str):
            severity = Severity(severity.lower())

        if severity == Severity.CRITICAL:
            symbol = self.CRITICAL_SYMBOL if self.severity_icons else ""
            css_class = "severity-critical"
        elif severity == Severity.WARNING:
            symbol = self.WARNING_SYMBOL if self.severity_icons else ""
            css_class = "severity-warning"
        else:
            symbol = self.INFO_SYMBOL if self.severity_icons else ""
            css_class = "severity-info"

        text = f"{symbol} {message}".strip() if symbol else message

        if html:
            return f'<div class="{css_class}">{text}</div>'
        return text

    def format_margin(
        self,
        value: float,
        limit: float,
        *,
        limit_type: Literal["upper", "lower"] = "upper",
        html: bool = False,
    ) -> str:
        """Format margin with color-coded indicator.

        Color coding:
        - Green: margin > 20%
        - Yellow: 10% < margin <= 20%
        - Red: margin <= 10%

        Args:
            value: Measured value.
            limit: Limit value.
            limit_type: Whether limit is upper or lower bound.
            html: Output as HTML with styling.

        Returns:
            Formatted margin string.

        References:
            REPORT-002: Visual Emphasis System
        """
        if limit_type == "upper":
            margin = limit - value
            margin_pct = (margin / limit * 100) if limit != 0 else 0
        else:
            margin = value - limit
            margin_pct = (margin / limit * 100) if limit != 0 else 0

        # Determine status
        if margin_pct > 20:
            status = "good"
            css_class = "pass"
            symbol = self.PASS_SYMBOL  # type: ignore[attr-defined]
        elif margin_pct > 10:
            status = "marginal"
            css_class = "warning"
            symbol = self.WARNING_SYMBOL
        elif margin_pct > 0:
            status = "tight"
            css_class = "warning"
            symbol = self.WARNING_SYMBOL
        else:
            status = "violation"
            css_class = "fail"
            symbol = self.CROSS_SYMBOL

        text = f"{symbol} margin: {margin_pct:.1f}% ({status})"

        if html:
            return f'<span class="{css_class}">{text}</span>'
        return text

    def create_callout_box(
        self,
        title: str,
        content: str,
        *,
        is_key_finding: bool = False,
    ) -> str:
        """Create a callout box for key findings.

        Args:
            title: Box title.
            content: Box content.
            is_key_finding: Style as key finding.

        Returns:
            HTML callout box string.

        References:
            REPORT-002: Visual Emphasis System
        """
        css_class = "callout key-finding" if is_key_finding else "callout"
        return f"""<div class="{css_class}">
<h4>{title}</h4>
<p>{content}</p>
</div>"""


@dataclass
class ExecutiveSummary:
    """Executive summary of analysis results.

    Attributes:
        overall_status: Pass/fail status.
        pass_count: Number of passing tests.
        total_count: Total number of tests.
        key_findings: List of key findings.
        critical_violations: List of critical violations.
        min_margin_pct: Minimum margin percentage.
        summary_text: Generated summary text.

    References:
        REPORT-004: Executive Summary Auto-Generation
    """

    overall_status: bool
    pass_count: int
    total_count: int
    key_findings: list[str] = field(default_factory=list)
    critical_violations: list[str] = field(default_factory=list)
    min_margin_pct: float | None = None
    summary_text: str = ""


def _extract_key_findings(
    results: dict[str, Any],
    critical_violations: list[Any],
    max_findings: int,
) -> tuple[list[str], float | None]:
    """Extract key findings from results."""
    key_findings: list[str] = []
    violations = results.get("violations", [])

    # Add violation summary
    if critical_violations:
        key_findings.append(
            f"{len(critical_violations)} critical violation(s) require immediate attention"
        )
    elif violations:
        key_findings.append(f"{len(violations)} violation(s) detected")

    # Add margin information
    margins = results.get("margins", [])
    min_margin = min(margins) if margins else results.get("min_margin")

    if min_margin is not None and min_margin < 20:
        status = "critical" if min_margin < 10 else "marginal"
        key_findings.append(f"Minimum margin is {min_margin:.1f}% ({status})")

    # Extract from failed measurements
    for name, meas in results.get("measurements", {}).items():
        if not meas.get("passed", True):
            key_findings.append(f"{name}: FAIL - {meas.get('message', 'violation')}")

    return key_findings[:max_findings], min_margin


def _build_summary_text(
    overall_status: bool,
    total_count: int,
    fail_count: int,
    critical_violations: list[Any],
    min_margin: float | None,
    key_findings: list[str],
    length: str,
) -> str:
    """Build the summary text."""
    parts: list[str] = []
    pass_count = total_count - fail_count

    # First sentence: overall status
    if overall_status and total_count > 0:
        parts.append(f"All {pass_count} tests passed.")
    elif overall_status:
        parts.append("Analysis completed successfully.")
    elif total_count > 0:
        pct = fail_count / total_count * 100
        parts.append(f"{fail_count} of {total_count} tests failed ({pct:.0f}% failure rate).")
    else:
        parts.append("Analysis completed with failures.")

    # Add critical violations
    if critical_violations:
        parts.append(f"Critical: {len(critical_violations)} violation(s) require immediate action.")

    # Add margin note
    if min_margin is not None and min_margin < 10:
        parts.append(f"Warning: Minimum margin is only {min_margin:.1f}%.")

    # Key findings (for detailed mode)
    if length == "detailed" and key_findings:
        parts.append("\nKey Findings:")
        parts.extend(f"  - {finding}" for finding in key_findings)

    return " ".join(parts)


def generate_executive_summary(
    results: dict[str, Any],
    *,
    max_findings: int = 5,
    length: Literal["short", "detailed"] = "short",
) -> ExecutiveSummary:
    """Generate executive summary from analysis results.

    Automatically extracts key findings, pass/fail status, and critical
    violations from analysis results.

    Args:
        results: Analysis results dictionary.
        max_findings: Maximum number of key findings to include.
        length: Summary length (short = 1 paragraph, detailed = 1 page).

    Returns:
        ExecutiveSummary with generated content.

    Example:
        >>> results = {"pass_count": 10, "total_count": 12, "violations": [...]}
        >>> summary = generate_executive_summary(results)
        >>> print(summary.summary_text)

    References:
        REPORT-004: Executive Summary Auto-Generation
    """
    # Extract basic counts
    pass_count = results.get("pass_count", 0)
    total_count = results.get("total_count", 0)
    fail_count = total_count - pass_count if total_count else 0
    overall_status = fail_count == 0

    # Extract violations
    violations = results.get("violations", [])
    critical_violations = [v for v in violations if v.get("severity", "").lower() == "critical"]

    # Extract key findings
    key_findings, min_margin = _extract_key_findings(results, critical_violations, max_findings)

    # Generate summary text
    summary_text = _build_summary_text(
        overall_status,
        total_count,
        fail_count,
        critical_violations,
        min_margin,
        key_findings,
        length,
    )

    return ExecutiveSummary(
        overall_status=overall_status,
        pass_count=pass_count,
        total_count=total_count,
        key_findings=key_findings,
        critical_violations=[str(v) for v in critical_violations],
        min_margin_pct=min_margin,
        summary_text=summary_text,
    )


def format_executive_summary_html(summary: ExecutiveSummary) -> str:
    """Format executive summary as HTML.

    Args:
        summary: ExecutiveSummary to format.

    Returns:
        HTML string.

    References:
        REPORT-004: Executive Summary Auto-Generation
    """
    emphasis = VisualEmphasis()

    status_html = emphasis.format_pass_fail(summary.overall_status, html=True)

    findings_html = ""
    if summary.key_findings:
        items = []
        for finding in summary.key_findings:
            css_class = "critical" if "critical" in finding.lower() else ""
            items.append(f'<li class="{css_class}">{finding}</li>')
        findings_html = f'<ul class="key-findings">{"".join(items)}</ul>'

    return f"""<div class="executive-summary">
<h2>Executive Summary</h2>
<p><strong>Overall Status:</strong> {status_html}</p>
<p>{summary.summary_text}</p>
{findings_html}
</div>"""


__all__ = [
    "ColorScheme",
    "ExecutiveSummary",
    "FormatStandards",
    "Severity",
    "VisualEmphasis",
    "format_executive_summary_html",
    "generate_executive_summary",
]
