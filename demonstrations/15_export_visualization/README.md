# Export and Visualization

**Share analysis results through multiple formats and visualization tools.**

This section contains 5 demonstrations showing how to export data in various formats, create timing diagrams, generate Wireshark dissectors, build comprehensive reports, and create visualization galleries. Essential for documenting and sharing reverse engineering findings.

---

## Prerequisites

Before running these demonstrations, ensure you have:

- **Completed Basic Analysis** - Run `demonstrations/02_basic_analysis/` first
- **Understanding of Data Formats** - CSV, JSON, HDF5 concepts
- **Familiarity with Visualization** - Basic plotting concepts
- **Protocol Knowledge** - Understanding of packet structures (for dissectors)

Check your setup:

```bash
python demonstrations/02_basic_analysis/01_waveform_measurements.py
# Should show successful measurements
```

---

## Demonstrations

| Demo                     | Time       | Difficulty       | Focus                                             |
| ------------------------ | ---------- | ---------------- | ------------------------------------------------- |
| 01_export_formats        | 10 min     | Beginner         | All export formats (CSV, JSON, HDF5, NPZ, MATLAB) |
| 02_wavedrom_timing       | 10 min     | Intermediate     | Timing diagrams with WaveDrom                     |
| 03_wireshark_dissectors  | 15 min     | Advanced         | Custom protocol dissectors                        |
| 04_report_generation     | 15 min     | Intermediate     | Automated analysis reports                        |
| 05_visualization_gallery | 10 min     | Beginner         | Complete visualization showcase                   |
| **Total**                | **60 min** | **Intermediate** | **Export and visualization**                      |

---

## Learning Path

Complete these demonstrations in order for comprehensive coverage:

```
01_export_formats.py    02_wavedrom_timing.py   03_wireshark_dissectors.py
         ↓                       ↓                          ↓
All export formats      Timing diagrams         Protocol dissectors
CSV/JSON/HDF5/MAT      WaveDrom integration    Wireshark plugins
         ↓                       ↓                          ↓
04_report_generation.py    05_visualization_gallery.py
         ↓                              ↓
Automated reports              Complete gallery
PDF/HTML/Markdown              All visualizations
```

### Estimated Time: 60 minutes

---

## Key Concepts

This section teaches:

1. **Multi-Format Export** - CSV, JSON, HDF5, NPZ, MATLAB with metadata preservation
2. **Timing Diagrams** - WaveDrom integration for digital signal visualization
3. **Protocol Dissectors** - Wireshark plugin generation for custom protocols
4. **Report Generation** - Automated PDF, HTML, and Markdown reports
5. **Visualization Gallery** - Complete showcase of all visualization capabilities

---

## Running Demonstrations

### Option 1: Run Individual Demo

```bash
# From the project root
python demonstrations/15_export_visualization/01_export_formats.py

# Or from the demo directory
cd demonstrations/15_export_visualization
python 01_export_formats.py
```

### Option 2: Run All Export/Visualization Demos

```bash
# From the project root
python demonstrations/15_export_visualization/01_export_formats.py && \
python demonstrations/15_export_visualization/02_wavedrom_timing.py && \
python demonstrations/15_export_visualization/03_wireshark_dissectors.py && \
python demonstrations/15_export_visualization/04_report_generation.py && \
python demonstrations/15_export_visualization/05_visualization_gallery.py
```

### Option 3: Validate All Demonstrations

```bash
# From the project root
python demonstrations/validate_all.py
```

---

## What You'll Learn

### Demo 01: Export Formats

**File**: `01_export_formats.py`

**Demonstrates**:

- CSV export with metadata preservation
- JSON export for structured data
- HDF5 export for large datasets
- NPZ export for NumPy arrays
- MATLAB export for .mat files
- Format comparison and best practices

**What you'll do**:

1. Export the same signal to all supported formats
2. Compare format features and use cases
3. Preserve metadata across different formats
4. Choose the right format for your needs

**Capabilities**:

- `oscura.exporters.csv` - CSV export with headers
- `oscura.exporters.json` - Structured JSON export
- `oscura.exporters.hdf5` - Large dataset export
- `oscura.exporters.npz` - NumPy array export
- `oscura.exporters.mat` - MATLAB compatibility

**Related Demos**:

- `15_export_visualization/04_report_generation.py` - Report generation
- `01_data_loading/01_oscilloscopes.py` - Loading data

---

### Demo 02: WaveDrom Timing Diagrams

**File**: `02_wavedrom_timing.py`

**Demonstrates**:

- WaveDrom timing diagram generation
- Digital signal visualization
- Protocol timing diagrams
- Multi-channel timing relationships

**What you'll do**:

1. Generate WaveDrom JSON from digital signals
2. Create timing diagrams for protocols
3. Visualize multi-channel relationships
4. Export timing diagrams for documentation

**Capabilities**:

- WaveDrom JSON generation
- Digital signal timing visualization
- Multi-channel synchronization
- Export to SVG/PNG formats

**Related Demos**:

- `03_protocol_decoding/01_serial_comprehensive.py` - Protocol decoding
- `15_export_visualization/04_report_generation.py` - Including diagrams in reports

---

### Demo 03: Wireshark Dissector Generation

**File**: `03_wireshark_dissectors.py`

**Demonstrates**:

- Automatic Wireshark dissector generation
- Custom protocol plugin creation
- PCAP file generation from decoded packets
- Wireshark integration workflow

**What you'll do**:

1. Generate Wireshark dissectors for custom protocols
2. Create PCAP files from decoded data
3. Load custom protocols in Wireshark
4. Analyze protocol data with Wireshark tools

**Capabilities**:

- Lua dissector generation
- PCAP file creation
- Protocol field definitions
- Wireshark plugin packaging

**Related Demos**:

- `03_protocol_decoding/01_serial_comprehensive.py` - Protocol decoding
- `06_reverse_engineering/01_unknown_protocol.py` - Protocol RE

---

### Demo 04: Report Generation

**File**: `04_report_generation.py`

**Demonstrates**:

- Automated PDF report generation
- HTML report creation
- Markdown documentation
- Including plots, tables, and analysis results

**What you'll do**:

1. Generate comprehensive PDF analysis reports
2. Create interactive HTML reports
3. Build Markdown documentation
4. Include visualizations and tables

**Capabilities**:

- PDF report generation (LaTeX/ReportLab)
- HTML report creation
- Markdown documentation
- Plot and table embedding

**Related Demos**:

- `15_export_visualization/01_export_formats.py` - Data export
- `15_export_visualization/05_visualization_gallery.py` - Visualizations

---

### Demo 05: Visualization Gallery

**File**: `05_visualization_gallery.py`

**Demonstrates**:

- Complete showcase of all visualization types
- Time domain plots (waveforms, traces)
- Frequency domain plots (FFT, spectrograms)
- Statistical plots (histograms, scatter)
- Protocol visualizations (packets, timing)

**What you'll do**:

1. See all available visualization types
2. Learn when to use each visualization
3. Customize plots for your needs
4. Export visualizations for documentation

**Capabilities**:

- Time domain visualization
- Frequency domain visualization
- Statistical visualization
- Protocol visualization
- Custom plot styling

**Related Demos**:

- `02_basic_analysis/01_waveform_measurements.py` - Basic plots
- `04_advanced_analysis/01_spectral_analysis.py` - Spectrograms
- `15_export_visualization/04_report_generation.py` - Including in reports

---

## Troubleshooting

### "Export failed: permission denied"

**Solution**: Ensure write permissions for the output directory:

```bash
# Check permissions
ls -la output/

# Create output directory with proper permissions
mkdir -p output/exports
chmod 755 output/exports
```

### "WaveDrom output is empty"

**Solution**: Ensure digital signals have state transitions:

```python
# Digital signal must have high/low states
digital_trace = DigitalTrace(
    data=np.array([0, 0, 1, 1, 0, 1, 0]),  # Clear transitions
    metadata=metadata
)
```

### "Wireshark dissector not loading"

**Solution**: Check Lua syntax and install location:

```bash
# Validate Lua syntax
luac -p dissector.lua

# Install to correct Wireshark plugins directory
cp dissector.lua ~/.local/lib/wireshark/plugins/
```

### "Report generation requires additional dependencies"

**Solution**: Install optional report generation packages:

```bash
# For PDF reports
pip install reportlab

# For LaTeX-based reports
sudo apt-get install texlive-latex-base

# For HTML reports (usually included)
pip install jinja2
```

---

## Next Steps

### If You Want to...

| Goal                              | Next Demo                                                 | Path               |
| --------------------------------- | --------------------------------------------------------- | ------------------ |
| Build complete analysis workflows | `16_complete_workflows/01_protocol_discovery.py`          | Complete workflows |
| Generate test signals             | `17_signal_generation/01_signal_builder_comprehensive.py` | Signal generation  |
| Perform comparison testing        | `18_comparison_testing/01_golden_reference.py`            | Comparison testing |
| Share findings with team          | Use report generation                                     | Documentation      |

### Recommended Next Sections

1. **Complete Workflows** (16_complete_workflows/)
   - End-to-end analysis examples
   - Production-ready workflows
   - Real-world case studies

2. **Signal Generation** (17_signal_generation/)
   - Test signal creation
   - Protocol generation
   - Impairment simulation

3. **Comparison Testing** (18_comparison_testing/)
   - Golden reference testing
   - Limit testing
   - Mask testing

---

## Understanding Export and Visualization

### Format Selection Guide

Choose the right export format for your use case:

| Format | Best For                          | Metadata | Size   | Tools              |
| ------ | --------------------------------- | -------- | ------ | ------------------ |
| CSV    | Spreadsheet analysis, simple data | Limited  | Medium | Excel, Python      |
| JSON   | Structured data, web APIs         | Full     | Medium | JavaScript, Python |
| HDF5   | Large datasets, scientific        | Full     | Small  | MATLAB, Python     |
| NPZ    | NumPy workflows                   | Limited  | Small  | Python (NumPy)     |
| MATLAB | MATLAB analysis                   | Full     | Medium | MATLAB             |

### Visualization Types

Different visualizations for different insights:

1. **Time Domain** - Waveforms, traces, digital signals
2. **Frequency Domain** - FFT, spectrograms, power spectral density
3. **Statistical** - Histograms, scatter plots, distributions
4. **Protocol** - Packet diagrams, timing diagrams, state machines

### Report Components

Comprehensive reports include:

- **Executive Summary** - High-level findings
- **Methodology** - Analysis approach
- **Results** - Measurements and observations
- **Visualizations** - Plots, diagrams, tables
- **Conclusions** - Interpretations and recommendations
- **Appendices** - Raw data, detailed calculations

---

## Best Practices

### Export Strategy

1. **Preserve Metadata** - Always export with metadata when possible
2. **Choose Appropriate Format** - Match format to intended use
3. **Document Units** - Include units in column headers/field names
4. **Version Exports** - Use timestamps or version numbers in filenames

### Visualization Design

1. **Clear Labels** - Always label axes with units
2. **Appropriate Scale** - Use logarithmic scale when needed
3. **Color Contrast** - Ensure plots are readable in grayscale
4. **Legends** - Include legends for multi-trace plots

### Report Quality

1. **Reproducibility** - Include enough detail to reproduce analysis
2. **Clear Structure** - Use sections and subsections logically
3. **Visual Quality** - Use high-resolution plots (300 DPI minimum)
4. **Peer Review** - Have reports reviewed before sharing

---

## Advanced Techniques

### Automated Report Pipelines

Create repeatable report generation workflows:

```python
# Define report template
report = AnalysisReport(
    title="Protocol Analysis",
    template="protocol_template.md"
)

# Add sections programmatically
report.add_section("Overview", overview_text)
report.add_plot("Waveform", waveform_plot)
report.add_table("Measurements", measurements_df)

# Generate multiple formats
report.export_pdf("report.pdf")
report.export_html("report.html")
report.export_markdown("report.md")
```

### Custom Wireshark Dissectors

Advanced dissector features:

```python
# Custom field formatting
dissector.add_field(
    name="timestamp",
    type="uint32",
    format="hex",  # Display as hexadecimal
    description="Packet timestamp"
)

# Conditional field parsing
dissector.add_conditional_field(
    name="payload",
    condition=lambda pkt: pkt.type == 0x01,
    parser=custom_payload_parser
)
```

### Interactive Visualizations

Create interactive HTML plots:

```python
# Enable interactive features
plot = create_plot(
    trace,
    interactive=True,
    zoom=True,
    pan=True,
    hover_info=True
)

# Export as standalone HTML
plot.export_html("interactive_plot.html")
```

---

## Tips for Success

### Metadata Preservation

Always include metadata in exports:

```python
# Bad: Data only
export_csv(trace.data, "signal.csv")

# Good: Data + metadata
export_csv(
    trace,
    "signal.csv",
    include_metadata=True,
    metadata_section="header"
)
```

### Format Conversion

Convert between formats while preserving metadata:

```python
# Load from one format
trace = load_csv("input.csv")

# Export to another format with metadata
export_hdf5(trace, "output.h5", preserve_metadata=True)
```

### Visualization Customization

Customize plots for professional reports:

```python
plot = create_waveform_plot(
    trace,
    title="Power Supply Ripple Analysis",
    xlabel="Time (ms)",
    ylabel="Voltage (V)",
    style="publication",  # High-quality styling
    dpi=300  # Print quality
)
```

---

## Format-Specific Details

### CSV Export

Best practices for CSV export:

```python
# Include units in headers
export_csv(
    trace,
    "signal.csv",
    headers=["Time (s)", "Voltage (V)"],
    metadata_rows=5  # Reserve top rows for metadata
)
```

### HDF5 Export

Organize large datasets efficiently:

```python
# Create hierarchical structure
export_hdf5(
    traces,
    "dataset.h5",
    structure={
        "/raw_data/": raw_traces,
        "/filtered/": filtered_traces,
        "/measurements/": measurement_results
    }
)
```

### MATLAB Export

Ensure MATLAB compatibility:

```python
# Use MATLAB-compatible data types
export_mat(
    trace,
    "signal.mat",
    variable_name="oscura_trace",  # Valid MATLAB identifier
    format="v7.3"  # Modern MATLAB format
)
```

---

## Summary

The Export and Visualization section covers:

| Demo                     | Focus               | Outcome                      |
| ------------------------ | ------------------- | ---------------------------- |
| 01_export_formats        | All export formats  | Multi-format data export     |
| 02_wavedrom_timing       | Timing diagrams     | Digital signal visualization |
| 03_wireshark_dissectors  | Protocol dissectors | Wireshark integration        |
| 04_report_generation     | Automated reports   | PDF/HTML/Markdown reports    |
| 05_visualization_gallery | Complete showcase   | All visualization types      |

After completing these 60-minute demonstrations, you'll understand:

- How to export data in all supported formats (CSV, JSON, HDF5, NPZ, MATLAB)
- How to create timing diagrams with WaveDrom
- How to generate Wireshark dissectors for custom protocols
- How to build automated analysis reports in multiple formats
- How to use all available visualization types effectively

**Ready to start?** Run this to explore export formats:

```bash
python demonstrations/15_export_visualization/01_export_formats.py
```

Happy documenting and sharing your findings!
