# Integration Demonstrations

This directory contains demonstrations of Oscura integration patterns with external systems, tools, and workflows.

## Demonstrations

### 01_cli_usage.py - Command-Line Interface Integration

**What it demonstrates:**

- Oscura CLI commands (characterize, decode, batch)
- Building custom CLI tools with argparse
- Batch file processing with progress tracking
- Multiple output formats (JSON, CSV, HTML, table)
- Logging configuration for CLI tools
- Error handling with proper exit codes

**Key capabilities:**

- `oscura.cli.main.format_output()` - Format results for different outputs
- `oscura.cli.batch.batch()` - Batch processing command
- `argparse` integration patterns
- Progress bar implementation

**When to use this:**

- Building command-line tools for signal processing
- Automating batch analysis workflows
- Creating deployment pipelines
- Integrating Oscura into shell scripts

**Example usage:**

```bash
python demonstrations/11_integration/01_cli_usage.py
```

---

### 02_jupyter_notebooks.py - Jupyter Notebook Integration

**What it demonstrates:**

- IPython magic commands (%oscura, %%analyze)
- Rich HTML display for traces and measurements
- Inline waveform and spectrum visualization
- Interactive widgets for parameter exploration
- Notebook-friendly output formats

**Key capabilities:**

- `oscura.jupyter.magic.OscuraMagics` - Magic commands
- `oscura.jupyter.display.TraceDisplay` - Rich trace display
- `oscura.jupyter.display.MeasurementDisplay` - Rich measurement display
- `oscura.jupyter.display.display_trace()` - Display convenience function
- `oscura.jupyter.display.display_spectrum()` - Inline spectrum plots

**When to use this:**

- Interactive signal exploration
- Creating analysis notebooks
- Teaching and documentation
- Rapid prototyping and experimentation

**Example usage:**

```python
# In Jupyter notebook
%load_ext oscura
%oscura load capture.wfm
%oscura measure rise_time fall_time

from oscura.jupyter import display_trace, display_measurements
display_trace(trace)
```

---

### 03_llm_integration.py - LLM/AI Integration

**What it demonstrates:**

- Structured JSON output for LLM consumption
- Natural language result summaries
- Question-answering format over analysis results
- Claude/ChatGPT integration patterns
- Semantic search context generation

**Key capabilities:**

- Structured output formatting
- Natural language summary generation
- Q&A pair creation
- Protocol detection for AI interpretation
- Embedding-ready document creation

**When to use this:**

- AI-assisted signal analysis
- Building chatbots for hardware troubleshooting
- Semantic search over signal data
- Automated report generation

**Example integration:**

```python
import anthropic
import oscura as osc

trace = osc.load("signal.wfm")
measurements = osc.measure(trace)

# Format for LLM
analysis_json = json.dumps({
    "signal_info": {...},
    "measurements": measurements
})

# Send to Claude
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": f"Analyze: {analysis_json}"}]
)
```

---

### 04_configuration_files.py - Configuration-Driven Workflows

**What it demonstrates:**

- YAML/JSON configuration loading
- Configuration schema validation
- Default values and override hierarchies
- Environment variable substitution
- Multi-stage pipeline configuration
- Config-driven analysis execution

**Key capabilities:**

- `oscura.config.load_config()` - Load configuration files
- `oscura.config.validate_against_schema()` - Schema validation
- `oscura.config.load_pipeline()` - Pipeline configuration
- Configuration merging and overrides

**When to use this:**

- Reproducible analysis workflows
- Managing complex analysis pipelines
- Environment-specific configurations
- Batch processing with different parameters

**Example configuration:**

```yaml
# config.yaml
version: '1.0'

signal:
  sample_rate: 100000
  channel: CH1

analysis:
  measurements:
    - frequency
    - amplitude
    - rise_time

output:
  format: json
  save_plots: true
```

---

### 05_hardware_integration.py - Hardware Data Acquisition

**What it demonstrates:**

- Hardware abstraction layer design
- Mock hardware sources for testing
- Oscilloscope simulation
- Logic analyzer simulation
- Hardware factory pattern
- Error handling and retry logic
- Real-time streaming acquisition
- Hardware configuration management

**Key capabilities:**

- `HardwareSource` abstract interface
- `MockOscilloscope` - Simulated oscilloscope
- `MockLogicAnalyzer` - Simulated logic analyzer
- `HardwareFactory` - Factory pattern for hardware creation
- Streaming acquisition patterns

**When to use this:**

- Integrating with real hardware (SocketCAN, Saleae, PyVISA)
- Building data acquisition systems
- Real-time signal monitoring
- Hardware abstraction for testing

**Real hardware adaptation:**

```python
# Adapt MockOscilloscope to real hardware
import pyvisa

class PyVISAOscilloscope(HardwareSource):
    def __init__(self, address: str):
        self.rm = pyvisa.ResourceManager()
        self.instrument = None
        self.address = address

    def connect(self) -> bool:
        self.instrument = self.rm.open_resource(self.address)
        return True

    def acquire(self, duration: float) -> WaveformTrace:
        # Real PyVISA acquisition
        self.instrument.write(":WAV:SOUR CHAN1")
        self.instrument.write(":WAV:MODE NORM")
        data = self.instrument.query_binary_values(":WAV:DATA?")
        ...
```

---

## Integration Patterns

### CLI Integration

- Use `format_output()` for consistent formatting
- Implement progress tracking for batch operations
- Handle errors gracefully with exit codes
- Support multiple output formats

### Jupyter Integration

- Load extension with `%load_ext oscura`
- Use magic commands for quick operations
- Rich displays for better visualization
- Export to pandas DataFrames

### LLM Integration

- Structure output as JSON
- Generate natural language summaries
- Create Q&A pairs for training
- Use semantic search for retrieval

### Configuration Integration

- Use YAML for human-readable configs
- Validate against schemas early
- Support environment variables
- Provide sensible defaults

### Hardware Integration

- Define abstract interfaces
- Create mocks for testing
- Implement retry logic
- Handle disconnections gracefully

---

## Running the Demonstrations

Run all demonstrations:

```bash
cd demonstrations/11_integration
python 01_cli_usage.py
python 02_jupyter_notebooks.py
python 03_llm_integration.py
python 04_configuration_files.py
python 05_hardware_integration.py
```

Or validate all at once:

```bash
cd demonstrations
python validate_all.py
```

---

## Related Documentation

- **CLI Reference**: See `src/oscura/cli/` for CLI implementation
- **Jupyter Integration**: See `src/oscura/jupyter/` for magic commands
- **Configuration System**: See `src/oscura/config/` for config loading
- **API Documentation**: See `docs/` for complete API reference

---

## Common Integration Scenarios

### 1. Automated Testing Pipeline

```bash
# Process all test captures
oscura batch 'test_captures/*.wfm' --analysis characterize --save-summary results.csv

# Check for failures
grep ERROR results.csv && exit 1
```

### 2. Interactive Exploration

```python
# In Jupyter
%load_ext oscura
%oscura load signal.wfm
display_trace(trace)
%oscura measure
```

### 3. AI-Assisted Analysis

```python
# Format for LLM
results = analyze_signal(trace)
summary = generate_natural_language_summary(results)
send_to_claude(summary)
```

### 4. Config-Driven Pipeline

```yaml
# pipeline.yaml
stages:
  - type: load
  - type: filter
    params: { low_pass: 50000 }
  - type: analyze
  - type: export
```

### 5. Real-Time Monitoring

```python
# Stream from hardware
streamer = StreamingAcquisition(hardware)
streamer.start_stream(process_chunk)
```

---

## Next Steps

After completing these demonstrations:

1. **Build CLI tools** - Create custom analysis tools with argparse
2. **Create notebooks** - Build interactive analysis notebooks
3. **Integrate AI** - Add LLM-assisted analysis to workflows
4. **Automate pipelines** - Use configs for reproducible analysis
5. **Connect hardware** - Adapt patterns for real hardware integration

See the main demonstrations README.md for the complete demonstration roadmap.
