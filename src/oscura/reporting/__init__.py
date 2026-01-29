"""Report generation module for Oscura.

This module provides professional report generation including PDF/HTML
output, templates, formatting, and multi-format export.
"""

# Comprehensive Analysis Report API (CAR-001 through CAR-007)
from oscura.reporting.analyze import (
    UnsupportedFormatError,
    analyze,
)
from oscura.reporting.auto_report import (
    Report as AutoReport,
)
from oscura.reporting.auto_report import (
    ReportMetadata,
)
from oscura.reporting.auto_report import (
    generate_report as generate_auto_report,
)
from oscura.reporting.batch import (
    BatchReportResult,
    aggregate_batch_measurements,
    batch_report,
    generate_batch_report,
)
from oscura.reporting.chart_selection import (
    ChartType,
    auto_select_chart,
    get_axis_scaling,
    recommend_chart_with_reasoning,
)
from oscura.reporting.comparison import (
    compare_waveforms,
    generate_comparison_report,
)
from oscura.reporting.config import (
    ANALYSIS_CAPABILITIES,
    AnalysisConfig,
    AnalysisDomain,
    AnalysisError,
    AnalysisResult,
    DataOutputConfig,
    DomainConfig,
    InputType,
    ProgressCallback,
    ProgressInfo,
    get_available_analyses,
)
from oscura.reporting.core import (  # core.py module
    Report,
    ReportConfig,
    Section,
    generate_report,
)
from oscura.reporting.core_formats import (  # core_formats/ directory
    MultiFormatRenderer,
    detect_format_from_extension,
    render_all_formats,
)
from oscura.reporting.engine import (
    AnalysisEngine,
)
from oscura.reporting.enhanced_reports import (
    EnhancedReportGenerator,
)
from oscura.reporting.enhanced_reports import (
    ReportConfig as EnhancedReportConfig,
)
from oscura.reporting.export import (
    batch_export_formats,
    export_multiple_reports,
    export_report,
)
from oscura.reporting.formatting import (
    NumberFormatter,
    format_margin,
    format_pass_fail,
    format_value,
    format_with_context,
    format_with_locale,
    format_with_units,
)
from oscura.reporting.html import (
    generate_html_report,
    save_html_report,
)
from oscura.reporting.index import (
    IndexGenerator,
    TemplateEngine,
)
from oscura.reporting.multichannel import (
    generate_multichannel_report,
)
from oscura.reporting.output import OutputManager
from oscura.reporting.pdf import (
    generate_pdf_report,
    save_pdf_report,
)
from oscura.reporting.plots import (
    PLOT_REGISTRY,
    PlotGenerator,
    register_plot,
)
from oscura.reporting.pptx_export import (
    PPTXPresentation,
    PPTXSlide,
    export_pptx,
    generate_presentation_from_report,
)
from oscura.reporting.sections import (
    create_conclusions_section,
    create_executive_summary_section,
    create_measurement_results_section,
    create_methodology_section,
    create_plots_section,
    create_standard_report_sections,
    create_title_section,
    create_violations_section,
)
from oscura.reporting.standards import (
    ColorScheme,
    ExecutiveSummary,
    FormatStandards,
    Severity,
    VisualEmphasis,
    format_executive_summary_html,
    generate_executive_summary,
)
from oscura.reporting.summary_generator import (
    Finding,
    Summary,
    generate_summary,
)
from oscura.reporting.tables import (
    create_comparison_table,
    create_measurement_table,
    create_statistics_table,
    format_batch_summary_table,
)
from oscura.reporting.template_system import (
    ReportTemplate,
    TemplateSection,
    list_templates,
    load_template,
)

__all__ = [
    # Comprehensive Analysis Report API (CAR-001 through CAR-007)
    "ANALYSIS_CAPABILITIES",
    "PLOT_REGISTRY",
    "AnalysisConfig",
    "AnalysisDomain",
    "AnalysisEngine",
    "AnalysisError",
    "AnalysisResult",
    # Auto Report
    "AutoReport",
    # Batch (REPORT-009, RPT-003)
    "BatchReportResult",
    # Chart Selection (REPORT-028)
    "ChartType",
    # Standards (REPORT-001, REPORT-002, REPORT-004)
    "ColorScheme",
    "DataOutputConfig",
    "DomainConfig",
    # Enhanced Reports (Feature 6)
    "EnhancedReportConfig",
    "EnhancedReportGenerator",
    "ExecutiveSummary",
    # Summary Generation
    "Finding",
    "FormatStandards",
    "IndexGenerator",
    "InputType",
    # Multi-format (REPORT-010)
    "MultiFormatRenderer",
    # Formatting (REPORT-026)
    "NumberFormatter",
    "OutputManager",
    # PPTX Export (REPORT-023)
    "PPTXPresentation",
    "PPTXSlide",
    "PlotGenerator",
    "ProgressCallback",
    "ProgressInfo",
    # Core
    "Report",
    "ReportConfig",
    "ReportMetadata",
    # Templates (RPT-002)
    "ReportTemplate",
    "Section",
    "Severity",
    "Summary",
    "TemplateEngine",
    "TemplateSection",
    "UnsupportedFormatError",
    "VisualEmphasis",
    "aggregate_batch_measurements",
    "analyze",
    "auto_select_chart",
    # Export
    "batch_export_formats",
    "batch_report",
    # Comparison
    "compare_waveforms",
    # Tables
    "create_comparison_table",
    # Sections
    "create_conclusions_section",
    "create_executive_summary_section",
    "create_measurement_results_section",
    "create_measurement_table",
    "create_methodology_section",
    "create_plots_section",
    "create_standard_report_sections",
    "create_statistics_table",
    "create_title_section",
    "create_violations_section",
    # Multi-format (REPORT-010)
    "detect_format_from_extension",
    "export_multiple_reports",
    "export_pptx",
    "export_report",
    "format_batch_summary_table",
    "format_executive_summary_html",
    "format_margin",
    "format_pass_fail",
    "format_value",
    "format_with_context",
    "format_with_locale",
    "format_with_units",
    "generate_auto_report",
    "generate_batch_report",
    "generate_comparison_report",
    "generate_executive_summary",
    # HTML Generation
    "generate_html_report",
    # Multi-Channel
    "generate_multichannel_report",
    # PDF Generation
    "generate_pdf_report",
    "generate_presentation_from_report",
    "generate_report",
    "generate_summary",
    "get_available_analyses",
    "get_axis_scaling",
    "list_templates",
    "load_template",
    "recommend_chart_with_reasoning",
    "register_plot",
    # Multi-format (REPORT-010)
    "render_all_formats",
    "save_html_report",
    "save_pdf_report",
]
