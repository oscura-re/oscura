# Reporting API Reference

> **Version**: 0.4.0
> **Last Updated**: 2026-01-19

## Overview

Oscura provides professional report generation capabilities including PDF, HTML, and PowerPoint output, with automatic chart selection, executive summaries, and batch processing.

## Quick Start

```python
import oscura as osc
from oscura.reporting import generate_report, save_pdf_report, save_html_report

# Load and analyze
trace = osc.load("capture.wfm")

# Generate report
report = generate_report(trace, title="Signal Analysis Report")

# Save in multiple formats
save_pdf_report(report, "report.pdf")
save_html_report(report, "report.html")
```

## Core Report Generation

### `generate_report()`

Generate a comprehensive analysis report.

```python
from oscura.reporting import generate_report, Report, ReportConfig

# Basic usage
report = generate_report(trace, title="My Report")

# With configuration
config = ReportConfig(
    title="Signal Analysis",
    author="Engineering Team",
    include_plots=True,
    include_raw_data=False,
)
report = generate_report(trace, config=config)
```

**Parameters:**

|Parameter|Type|Description|
|---|---|---|
|`trace`|`WaveformTrace`|Input waveform trace|
|`title`|`str`|Report title|
|`config`|`ReportConfig`|Optional configuration|

**Returns:** `Report` object containing sections, data, and metadata.

### `Report` Class

```python
from oscura.reporting import Report, Section

report = Report(
    title="Analysis Report",
    sections=[
        Section(title="Executive Summary", content="..."),
        Section(title="Measurements", content="..."),
    ],
    metadata={"author": "Oscura", "date": "2026-01-02"}
)

# Access report data
print(report.title)
for section in report.sections:
    print(f"  {section.title}")
```

### `ReportConfig` Class

```python
from oscura.reporting import ReportConfig

config = ReportConfig(
    title="Signal Analysis Report",
    author="Engineering Team",
    company="ACME Corp",
    include_plots=True,
    include_raw_data=False,
    include_statistics=True,
    plot_format="png",
    plot_dpi=150,
)
```

## Output Formats

### PDF Generation

```python
from oscura.reporting import generate_pdf_report, save_pdf_report

# Generate and save
report = generate_report(trace)
save_pdf_report(report, "output.pdf")

# Or generate PDF directly
pdf_bytes = generate_pdf_report(report)
with open("output.pdf", "wb") as f:
    f.write(pdf_bytes)
```

### HTML Generation

```python
from oscura.reporting import generate_html_report, save_html_report

# Generate and save
report = generate_report(trace)
save_html_report(report, "output.html")

# Or get HTML string
html_content = generate_html_report(report)
```

### PowerPoint Export

```python
from oscura.reporting import export_pptx, generate_presentation_from_report

# From report
report = generate_report(trace)
pptx = generate_presentation_from_report(report)
pptx.save("presentation.pptx")

# Direct export
export_pptx(trace, "presentation.pptx", title="Signal Analysis")
```

### Multi-Format Export

```python
from oscura.reporting import export_report, batch_export_formats

# Single format
export_report(report, "output.pdf", format="pdf")

# Multiple formats at once
batch_export_formats(
    report,
    base_path="output",
    formats=["pdf", "html", "pptx"]
)
# Creates: output.pdf, output.html, output.pptx
```

## Comprehensive Analysis Reports

### `analyze()` Function

Run comprehensive analysis and generate results.

```python
from oscura.reporting import analyze, AnalysisDomain

# Full analysis
results = analyze(trace)

# Specific domains only
results = analyze(
    trace,
    domains=[
        AnalysisDomain.WAVEFORM,
        AnalysisDomain.SPECTRAL,
        AnalysisDomain.DIGITAL,
    ]
)

# Access results
print(results['waveform']['frequency'])
print(results['spectral']['thd'])
```

### Analysis Domains

```python
from oscura.reporting import AnalysisDomain

# Available domains
AnalysisDomain.WAVEFORM     # Basic waveform measurements
AnalysisDomain.SPECTRAL     # FFT, PSD, harmonics
AnalysisDomain.DIGITAL      # Digital timing, edges
AnalysisDomain.TIMING       # Setup/hold, skew
AnalysisDomain.STATISTICS   # Statistical analysis
AnalysisDomain.PATTERNS     # Pattern discovery
AnalysisDomain.JITTER       # Jitter analysis
AnalysisDomain.EYE          # Eye diagram
AnalysisDomain.PROTOCOLS    # Protocol decoding
AnalysisDomain.ENTROPY      # Entropy analysis
AnalysisDomain.INFERENCE    # Protocol inference
```

### Analysis Configuration

```python
from oscura.reporting import AnalysisConfig, DomainConfig

config = AnalysisConfig(
    domains={
        AnalysisDomain.WAVEFORM: DomainConfig(enabled=True),
        AnalysisDomain.SPECTRAL: DomainConfig(
            enabled=True,
            parameters={"window": "hanning", "nfft": 4096}
        ),
    },
    progress_callback=lambda info: print(f"Progress: {info.percent}%"),
)

results = analyze(trace, config=config)
```

## Report Sections

### Standard Sections

```python
from oscura.reporting import (
    create_standard_report_sections,
    create_title_section,
    create_executive_summary_section,
    create_measurement_results_section,
    create_methodology_section,
    create_conclusions_section,
    create_plots_section,
    create_violations_section,
)

# Create all standard sections
sections = create_standard_report_sections(results, config)

# Or create individual sections
title = create_title_section("Signal Analysis", author="Team")
summary = create_executive_summary_section(results)
measurements = create_measurement_results_section(results)
```

### Executive Summary

```python
from oscura.reporting import generate_executive_summary, ExecutiveSummary

summary = generate_executive_summary(results)
print(summary.key_findings)
print(summary.recommendations)
print(summary.pass_fail_status)
```

## Formatting

### Number Formatting

```python
from oscura.reporting import format_value, format_with_units, NumberFormatter

# Auto-format with SI prefixes
formatted = format_value(1.234e-9)  # "1.234 ns"
formatted = format_with_units(1500000, "Hz")  # "1.5 MHz"

# Custom formatter
formatter = NumberFormatter(precision=3, use_si_prefix=True)
formatted = formatter.format(1234567.89, "V")  # "1.235 MV"
```

### Locale-Aware Formatting

```python
from oscura.reporting import format_with_locale

# German locale (comma decimal separator)
formatted = format_with_locale(1234.56, locale="de_DE")  # "1.234,56"

# French locale
formatted = format_with_locale(1234.56, locale="fr_FR")  # "1 234,56"
```

### Pass/Fail Formatting

```python
from oscura.reporting import format_pass_fail, format_margin

# Pass/fail status
status = format_pass_fail(measured=4.8, limit=5.0, limit_type="max")
# Returns: {"status": "PASS", "margin": "4.0%", "color": "green"}

# Margin calculation
margin = format_margin(measured=4.8, limit=5.0)  # "4.0%"
```

## Tables

### Measurement Tables

```python
from oscura.reporting import create_measurement_table, create_statistics_table

# Measurement table
table = create_measurement_table(results['waveform'])
# Returns formatted table data

# Statistics table
stats_table = create_statistics_table(results['statistics'])
```

### Comparison Tables

```python
from oscura.reporting import create_comparison_table

table = create_comparison_table(
    results_a=results1,
    results_b=results2,
    labels=["Before", "After"]
)
```

## Batch Processing

### Batch Reports

```python
from oscura.reporting import batch_report, generate_batch_report, BatchReportResult

# Process multiple files
files = ["file1.wfm", "file2.wfm", "file3.wfm"]
results = batch_report(files)

for result in results:
    print(f"{result.filename}: {result.status}")
    if result.report:
        save_pdf_report(result.report, f"{result.filename}.pdf")

# Generate aggregate report
aggregate = generate_batch_report(results)
save_pdf_report(aggregate, "batch_summary.pdf")
```

### Batch Aggregation

```python
from oscura.reporting import aggregate_batch_measurements

# Aggregate measurements across files
aggregated = aggregate_batch_measurements(results)
print(f"Average frequency: {aggregated['frequency']['mean']}")
print(f"Frequency std dev: {aggregated['frequency']['std']}")
```

## Comparison Reports

```python
from oscura.reporting import compare_waveforms, generate_comparison_report

# Compare two waveforms
comparison = compare_waveforms(trace_a, trace_b)
print(f"Amplitude difference: {comparison['amplitude_diff']}")
print(f"Phase shift: {comparison['phase_shift']}")

# Generate comparison report
report = generate_comparison_report(
    trace_a, trace_b,
    labels=["Reference", "Device Under Test"]
)
save_pdf_report(report, "comparison.pdf")
```

## Multi-Channel Reports

```python
from oscura.reporting import generate_multichannel_report

channels = osc.load_all_channels("multi_channel.wfm")
report = generate_multichannel_report(channels)
save_pdf_report(report, "multichannel_report.pdf")
```

## Chart Selection

### Automatic Chart Selection

```python
from oscura.reporting import auto_select_chart, ChartType

# Auto-select best chart type
chart_type = auto_select_chart(data, measurement_type="frequency")
# Returns: ChartType.LINE, ChartType.HISTOGRAM, etc.
```

### Chart Recommendations

```python
from oscura.reporting import recommend_chart_with_reasoning

recommendation = recommend_chart_with_reasoning(data, context="time_series")
print(f"Recommended: {recommendation['chart_type']}")
print(f"Reasoning: {recommendation['reasoning']}")
```

### Axis Scaling

```python
from oscura.reporting import get_axis_scaling

scaling = get_axis_scaling(data)
print(f"X range: {scaling['x_min']} to {scaling['x_max']}")
print(f"Y range: {scaling['y_min']} to {scaling['y_max']}")
print(f"Log scale recommended: {scaling['use_log']}")
```

## Templates

### Using Templates

```python
from oscura.reporting import load_template, list_templates, ReportTemplate

# List available templates
templates = list_templates()
for name in templates:
    print(name)

# Load and use template
template = load_template("standard")
report = template.generate(results)
```

### Custom Templates

```python
from oscura.reporting import ReportTemplate, TemplateSection

template = ReportTemplate(
    name="custom",
    sections=[
        TemplateSection(name="summary", required=True),
        TemplateSection(name="measurements", required=True),
        TemplateSection(name="plots", required=False),
    ],
    style={
        "font_family": "Arial",
        "primary_color": "#0066cc",
    }
)
```

## Visual Standards

### Formatting Standards

```python
from oscura.reporting import FormatStandards, ColorScheme, Severity

# Access standard colors
print(ColorScheme.PASS)    # Green
print(ColorScheme.FAIL)    # Red
print(ColorScheme.WARNING) # Yellow

# Severity levels
Severity.CRITICAL
Severity.WARNING
Severity.INFO
```

### Visual Emphasis

```python
from oscura.reporting import VisualEmphasis

# Apply emphasis based on value
emphasis = VisualEmphasis.for_value(
    value=95.5,
    threshold_warning=90,
    threshold_critical=80
)
print(emphasis.color)
print(emphasis.icon)
```

## Progress Tracking

```python
from oscura.reporting import analyze, ProgressCallback, ProgressInfo

def on_progress(info: ProgressInfo):
    print(f"[{info.percent:3.0f}%] {info.current_domain}: {info.message}")

results = analyze(
    trace,
    progress_callback=on_progress
)
```

## Summary Generation

```python
from oscura.reporting import generate_summary, Summary, Finding

summary = generate_summary(results)

print(f"Overall status: {summary.status}")
print(f"Key findings:")
for finding in summary.findings:
    print(f"  [{finding.severity}] {finding.message}")
```

## Analysis Engine

For advanced use cases, use the `AnalysisEngine` directly:

```python
from oscura.reporting import AnalysisEngine

engine = AnalysisEngine()

# Run analysis
results = engine.analyze(trace, domains=[AnalysisDomain.WAVEFORM])

# Get available analyses
available = engine.get_available_analyses()
```

## See Also

- [Loader API](loader.md) - Data loading
- [Analysis API](analysis.md) - Analysis functions
- [Export API](export.md) - Data export
- [Visualization API](visualization.md) - Plotting utilities
