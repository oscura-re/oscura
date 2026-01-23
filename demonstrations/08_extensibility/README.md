# Extensibility and Plugin Development

**Extend Oscura with custom measurements, algorithms, and plugins.**

This section contains 6 demonstrations designed to teach you how to extend Oscura's functionality through its plugin system, custom measurements, and algorithm registration. Perfect for users who need domain-specific analysis or want to integrate Oscura with proprietary workflows.

---

## Prerequisites

Before running these demonstrations, ensure you have:

- **Completed Getting Started** - Understanding of core types and basic measurements
- **Python 3.12+** - Required for Oscura
- **Basic Python development experience** - Understanding of classes, decorators, and type hints
- **Familiarity with Oscura measurements** - Complete `02_basic_analysis/01_waveform_measurements.py` first

Check your understanding:

```bash
# Should complete without errors
python demonstrations/00_getting_started/00_hello_world.py
python demonstrations/02_basic_analysis/01_waveform_measurements.py
```

---

## Learning Path

These demonstrations are designed to be completed **in order**. Each builds on concepts from the previous one:

```
01_plugin_basics.py → 02_custom_measurement.py → 03_custom_algorithm.py
        ↓                       ↓                           ↓
    Discover plugins      Register measurements      Register algorithms
    Inspect metadata      Use custom functions       Custom FFT/filters

        ↓                       ↓                           ↓
04_plugin_development.py → 05_measurement_registry.py → 06_plugin_templates.py
        ↓                       ↓                           ↓
   Full plugin lifecycle   Explore registry           Generate scaffolding
   Custom decoders         Dynamic invocation         Quick-start templates
```

### Estimated Time

| Demo                    | Time       | Difficulty                | Topics                              |
| ----------------------- | ---------- | ------------------------- | ----------------------------------- |
| 01_plugin_basics        | 10 min     | Intermediate              | Plugin discovery, metadata, loading |
| 02_custom_measurement   | 15 min     | Intermediate              | Custom measurements, registration   |
| 03_custom_algorithm     | 15 min     | Intermediate              | FFT, filters, algorithm registry    |
| 04_plugin_development   | 20 min     | Advanced                  | Full plugin lifecycle, decoders     |
| 05_measurement_registry | 10 min     | Intermediate              | Registry exploration, dynamic use   |
| 06_plugin_templates     | 10 min     | Intermediate              | Template generation, scaffolding    |
| **Total**               | **80 min** | **Intermediate-Advanced** | **Plugin system mastery**           |

---

## Demonstrations

### Demo 01: Plugin Basics

**File**: `01_plugin_basics.py`

**What it teaches**:

- Access the global plugin manager with `get_plugin_manager()`
- Discover available plugins by group with `list_plugins()`
- Load plugins from specific groups
- Inspect plugin metadata (name, version, capabilities, dependencies)
- Perform plugin health checking

**What you'll do**:

1. Query the plugin manager for all available plugins
2. List plugins organized by category (loaders, analyzers, decoders)
3. Inspect detailed metadata for specific plugins
4. Load a plugin and verify its capabilities
5. Check plugin health and compatibility

**Concepts introduced**:

- `PluginManager` - Central registry for all plugins
- `PluginMetadata` - Plugin information and capabilities
- Plugin groups (loaders, analyzers, decoders)
- Plugin discovery patterns
- Health checking and validation

**Why this matters**: Understanding the plugin architecture is essential before creating your own extensions. This demo shows how Oscura discovers and loads functionality dynamically.

---

### Demo 02: Custom Measurements

**File**: `02_custom_measurement.py`

**What it teaches**:

- Register custom measurements with `register_measurement()`
- Access the measurement registry with `get_measurement_registry()`
- List all available measurements with `list_measurements()`
- Create custom measurement functions
- Use custom measurements like built-in ones

**What you'll do**:

1. Define a custom measurement function (e.g., crest factor, duty cycle)
2. Register it with the measurement registry
3. Apply it to waveform data
4. List all measurements including your custom ones
5. Query measurement metadata programmatically

**Key concepts**:

- `MeasurementDefinition` - Metadata for custom measurements
- Registration patterns (decorators vs explicit registration)
- Type safety and validation
- Integration with existing workflow

**Why this matters**: Custom measurements let you extend Oscura with domain-specific analysis (automotive diagnostics, power quality, RF characterization) without modifying framework code.

---

### Demo 03: Custom Algorithms

**File**: `03_custom_algorithm.py`

**What it teaches**:

- Register custom algorithms with `register_algorithm()`
- Retrieve algorithms with `get_algorithm()`
- List algorithms by category with `get_algorithms()`
- Create custom FFT implementations
- Create custom filter algorithms
- Integrate custom analysis methods

**What you'll do**:

1. Implement a custom FFT algorithm (windowed, zero-padded)
2. Register it in the algorithm registry
3. Create a custom filter (Butterworth, Chebyshev)
4. Apply custom algorithms to signal data
5. Compare custom vs built-in algorithm performance

**Algorithm categories**:

- **FFT algorithms** - Custom frequency analysis methods
- **Filter algorithms** - Custom filtering and smoothing
- **Analysis algorithms** - Custom signal processing pipelines

**Why this matters**: Algorithm registration enables integration of specialized signal processing (GPU-accelerated FFT, proprietary filters, domain-specific transforms) seamlessly into Oscura workflows.

---

### Demo 04: Plugin Development

**File**: `04_plugin_development.py`

**What it teaches**:

- Complete plugin class structure and registration
- Creating custom loaders for proprietary formats
- Building custom analyzers for specific domains
- Implementing custom protocol decoders
- Plugin testing and validation workflows
- Documentation and metadata best practices

**What you'll do**:

1. Create a complete custom decoder plugin
2. Implement required plugin interface methods
3. Register the plugin with proper metadata
4. Test the plugin with sample data
5. Document capabilities and dependencies
6. Deploy the plugin for reuse

**Plugin types covered**:

- **Loaders** - Custom file format parsers
- **Analyzers** - Domain-specific analysis tools
- **Decoders** - Protocol decoders (UART, SPI, custom)

**Why this matters**: Full plugin development lets you package and distribute custom functionality as reusable modules, enabling team collaboration and workflow standardization.

---

### Demo 05: Measurement Registry

**File**: `05_measurement_registry.py`

**What it teaches**:

- Explore the measurement registry comprehensively
- Query measurements by category and tags
- Retrieve measurement metadata programmatically
- Invoke measurements dynamically by name
- Introspect measurement capabilities
- Build dynamic analysis pipelines

**What you'll do**:

1. List all registered measurements
2. Filter measurements by category (waveform, spectral, power)
3. Query metadata (units, description, parameters)
4. Invoke measurements dynamically using registry
5. Build a measurement suite from registry queries
6. Create auto-discovery workflows

**Use cases**:

- **Dynamic analysis** - Build measurement pipelines from configuration
- **Auto-discovery** - Find measurements matching criteria
- **Introspection** - Explore available capabilities programmatically
- **Documentation generation** - Auto-generate measurement catalogs

**Why this matters**: Registry introspection enables building GUI tools, configuration-driven analysis, and self-documenting workflows that adapt to available measurements.

---

### Demo 06: Plugin Templates

**File**: `06_plugin_templates.py`

**What it teaches**:

- Generate plugin scaffolding with `generate_plugin_template()`
- Use templates for different plugin types
- Understand generated file structure
- Customize templates for specific needs
- Template validation and best practices

**What you'll do**:

1. Generate a loader plugin template
2. Generate an analyzer plugin template
3. Generate a decoder plugin template
4. Explore generated file structure
5. Customize template for specific use case
6. Validate generated code structure

**Generated artifacts**:

- **Plugin class** - Complete implementation skeleton
- **Tests** - Unit test templates
- **Documentation** - Docstring templates
- **Metadata** - Version, capabilities, dependencies

**Why this matters**: Templates accelerate plugin development by providing complete, validated scaffolding with all required boilerplate, tests, and documentation structure.

---

## How to Run the Demos

### Option 1: Run Individual Demo

Run a single demo to learn a specific concept:

```bash
# From the project root
python demonstrations/08_extensibility/01_plugin_basics.py

# Or from the demo directory
cd demonstrations/08_extensibility
python 01_plugin_basics.py
```

Expected output: Plugin discovery results, metadata display, loading confirmation.

### Option 2: Run All Extensibility Demos

Run all six demos in sequence:

```bash
# From the project root
python demonstrations/08_extensibility/01_plugin_basics.py && \
python demonstrations/08_extensibility/02_custom_measurement.py && \
python demonstrations/08_extensibility/03_custom_algorithm.py && \
python demonstrations/08_extensibility/04_plugin_development.py && \
python demonstrations/08_extensibility/05_measurement_registry.py && \
python demonstrations/08_extensibility/06_plugin_templates.py
```

### Option 3: Validate All Demonstrations

Validate all demonstrations in the project:

```bash
# From the project root
python demonstrations/validate_all.py
```

This runs all demonstrations including extensibility and reports coverage.

---

## What You'll Learn

After completing this section, you will understand:

### Core Plugin Concepts

- **Plugin architecture** - How Oscura discovers and loads extensions
- **Plugin groups** - Loaders, analyzers, decoders organization
- **Plugin metadata** - Version, capabilities, dependencies
- **Health checking** - Validation and compatibility verification

### Custom Measurements

- **Registration patterns** - Decorator and explicit registration
- **Measurement definitions** - Metadata, units, parameters
- **Type safety** - Validation and error handling
- **Integration** - Using custom measurements like built-ins

### Algorithm Registry

- **Algorithm categories** - FFT, filters, analysis pipelines
- **Registration API** - `register_algorithm()`, `get_algorithm()`
- **Dynamic invocation** - Runtime algorithm selection
- **Performance** - Custom vs built-in comparison

### Plugin Development

- **Plugin structure** - Required methods and interfaces
- **Testing patterns** - Unit tests for plugins
- **Documentation** - Metadata and docstring requirements
- **Deployment** - Packaging and distribution

### Advanced Topics

- **Dynamic workflows** - Build analysis from configuration
- **Template generation** - Scaffolding for rapid development
- **Registry introspection** - Programmatic capability exploration
- **Best practices** - Design patterns for extensibility

---

## Common Issues and Solutions

### "ModuleNotFoundError" when importing plugin

**Solution**: Plugin modules must be importable. Ensure your plugin is:

1. In Python path: `sys.path.insert(0, str(Path(__file__).parent))`
2. Has `__init__.py` in all directories
3. Uses absolute imports for dependencies

```python
# Correct plugin import structure
from oscura.plugins import BasePlugin
from oscura.core.types import WaveformTrace
```

### Custom measurement not appearing in registry

**Solution**: Ensure you've registered the measurement before querying:

```python
# Register first
osc.register_measurement("my_measurement", my_function)

# Then query
measurements = osc.list_measurements()
assert "my_measurement" in measurements
```

### Plugin fails health check

**Solution**: Plugins must satisfy dependency requirements. Check:

1. All required dependencies are installed
2. Plugin version compatible with Oscura version
3. Required capabilities are implemented
4. Plugin metadata is complete

```python
# Verify plugin metadata
plugin = osc.load_plugin("my_plugin", group="analyzers")
print(plugin.metadata.dependencies)  # Check requirements
```

### Template generation fails

**Solution**: Ensure template output directory:

1. Exists and is writable
2. Doesn't contain existing files (avoid overwrite)
3. Has proper permissions

```python
import tempfile
from pathlib import Path

# Use temporary directory for testing
output_dir = Path(tempfile.mkdtemp())
osc.generate_plugin_template("my_plugin", output_dir=output_dir)
```

### Custom algorithm not invoked

**Solution**: Algorithm registration requires category specification:

```python
# Correct registration with category
osc.register_algorithm(
    name="my_fft",
    category="fft",  # Required
    function=my_fft_implementation
)

# Retrieve by category
fft_func = osc.get_algorithm("my_fft", category="fft")
```

---

## Next Steps: Where to Go After Extensibility

### If You Want to...

| Goal                                      | Next Demo                                   | Path                         |
| ----------------------------------------- | ------------------------------------------- | ---------------------------- |
| Apply custom measurements to batch data   | `09_batch_processing/01_parallel_batch.py`  | Batch → Parallel processing  |
| Build domain-specific analysis session    | `10_sessions/01_analysis_session.py`        | Sessions → Custom workflows  |
| Integrate custom decoders with protocols  | `03_protocol_decoding/06_auto_detection.py` | Protocols → Auto-detection   |
| Create quality checks with custom metrics | `12_quality_tools/02_quality_scoring.py`    | Quality → Custom scoring     |
| Deploy plugins for team use               | `11_integration/`                           | Integration → Team workflows |

### Recommended Learning Sequence

1. **Complete Extensibility** (this section)
   - Master plugin architecture
   - Create custom measurements
   - Develop reusable plugins

2. **Apply to Batch Processing** (09_batch_processing/)
   - Use custom measurements on multiple files
   - Parallel execution of custom analysis
   - Aggregate custom measurement results

3. **Integrate with Sessions** (10_sessions/)
   - Build domain-specific session types
   - Custom analysis workflows
   - Persistent custom analysis state

4. **Create Quality Tools** (12_quality_tools/)
   - Custom quality metrics
   - Domain-specific validation
   - Automated analysis recommendations

5. **Production Deployment** (11_integration/)
   - Package plugins for distribution
   - Team collaboration workflows
   - CI/CD integration

---

## Understanding Plugin Architecture

### Plugin Discovery

Oscura discovers plugins through:

1. **Entry points** - Python package entry points (pip install)
2. **Plugin directories** - Configurable plugin paths
3. **Dynamic registration** - Runtime `register_plugin()` calls

### Plugin Lifecycle

```
Discovery → Validation → Loading → Initialization → Use → Cleanup
    ↓           ↓           ↓            ↓          ↓       ↓
 Scan paths  Metadata   Import code   Configure  Execute  Unload
```

### Measurement Registry

Central catalog of all measurements:

- **Built-in measurements** - Amplitude, frequency, RMS, power, etc.
- **Custom measurements** - User-defined domain-specific metrics
- **Plugin measurements** - Measurements from loaded plugins
- **Dynamic measurements** - Generated at runtime

### Algorithm Categories

Algorithms organized by purpose:

| Category  | Purpose            | Examples                            |
| --------- | ------------------ | ----------------------------------- |
| fft       | Frequency analysis | Welch, periodogram, custom FFT      |
| filter    | Signal filtering   | Butterworth, Chebyshev, FIR, IIR    |
| analysis  | Signal processing  | Envelope, demodulation, correlation |
| detection | Event detection    | Edge, peak, threshold detection     |

---

## Best Practices

### Plugin Development

**DO**:

- Document all public methods with examples
- Include comprehensive tests (>80% coverage)
- Specify all dependencies explicitly
- Version your plugins semantically (semver)
- Validate inputs thoroughly

**DON'T**:

- Modify global state without cleanup
- Assume specific Oscura version without checking
- Skip metadata (breaks discovery)
- Use bare `except:` clauses (hides errors)

### Custom Measurements

**DO**:

- Return measurements with units
- Handle edge cases (empty data, NaN values)
- Document expected input types
- Provide usage examples in docstrings

**DON'T**:

- Modify input data (side effects)
- Return None without documentation
- Skip type hints (breaks introspection)
- Use ambiguous names

### Algorithm Registration

**DO**:

- Specify category explicitly
- Document performance characteristics
- Validate algorithm inputs
- Provide fallback for edge cases

**DON'T**:

- Assume specific NumPy version
- Skip error handling
- Ignore performance implications
- Create circular dependencies

---

## Resources

### In This Repository

- **`src/oscura/plugins/`** - Plugin base classes and interfaces
- **`src/oscura/measurement/`** - Measurement registry implementation
- **`src/oscura/algorithms/`** - Algorithm registry and built-ins
- **`tests/unit/test_plugins.py`** - Plugin test examples

### Example Plugins

- **Custom loader** - `examples/plugins/custom_loader.py`
- **Custom analyzer** - `examples/plugins/custom_analyzer.py`
- **Custom decoder** - `examples/plugins/custom_decoder.py`

### External Resources

- **[Python Entry Points](https://packaging.python.org/specifications/entry-points/)** - Plugin discovery mechanism
- **[Semantic Versioning](https://semver.org/)** - Plugin version standards
- **[Type Hints PEP 484](https://peps.python.org/pep-0484/)** - Type annotation guide

---

## Summary

The Extensibility section covers:

| Demo                    | Focus                     | Outcome                        |
| ----------------------- | ------------------------- | ------------------------------ |
| 01_plugin_basics        | Plugin discovery          | Understand plugin architecture |
| 02_custom_measurement   | Measurement registration  | Create custom measurements     |
| 03_custom_algorithm     | Algorithm registration    | Integrate custom algorithms    |
| 04_plugin_development   | Complete plugin lifecycle | Build full plugins             |
| 05_measurement_registry | Registry exploration      | Dynamic measurement use        |
| 06_plugin_templates     | Scaffolding generation    | Accelerate plugin development  |

After completing these 80-minute demonstrations, you'll be able to:

- Extend Oscura with custom functionality
- Create and distribute reusable plugins
- Build domain-specific analysis tools
- Integrate proprietary algorithms seamlessly
- Accelerate development with templates

**Ready to start?** Run this to understand the plugin system:

```bash
python demonstrations/08_extensibility/01_plugin_basics.py
```

Happy extending!
