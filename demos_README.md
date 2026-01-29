# Oscura Demonstrations Catalog

**Version**: 0.6.0 (Consolidated Structure)
**Last Updated**: 2026-01-29
**Total Demonstrations**: ~120 (target after consolidation)

---

## Quick Start

### Your First Demo

```bash
# Start here - validates your installation
python3 demos/00_getting_started/00_hello_world.py

# Understand data structures
python3 demos/00_getting_started/01_core_types.py

# Survey all capabilities
python3 demos/00_getting_started/02_supported_formats.py
```

### Running Demos

```bash
# Standard execution with validation
python3 demos/CATEGORY/demo_name.py

# Verbose output
python3 demos/CATEGORY/demo_name.py --verbose

# Skip validation (faster)
python3 demos/CATEGORY/demo_name.py --no-validate

# Use custom data file
python3 demos/CATEGORY/demo_name.py --data-file=/path/to/file
```

---

## Learning Paths

### Beginner Path (0-2 hours)

**Start here if you're new to Oscura**

1. `00_getting_started/00_hello_world.py` ⭐ **START HERE**
   - Your first Oscura workflow
   - Load → Measure → Analyze
   - ~5 minutes

2. `00_getting_started/01_core_types.py`
   - Understand WaveformTrace and DigitalTrace
   - Metadata and units
   - ~10 minutes

3. `01_data_loading/01_oscilloscopes.py`
   - Load data from oscilloscopes
   - Tektronix and Rigol formats
   - ~15 minutes

4. `02_basic_analysis/01_waveform_basics.py`
   - First measurements: amplitude, frequency, RMS
   - Understanding results
   - ~15 minutes

**Total**: ~45 minutes | **Prerequisites**: None | **Next**: Intermediate Path

### Intermediate Path (2-8 hours)

**For users comfortable with basics**

1. **Protocol Decoding Fundamentals** (1-2 hours)
   - `03_protocol_decoding/01_uart_basic.py` - UART serial communication
   - `03_protocol_decoding/02_spi_basic.py` - SPI bus analysis
   - `03_protocol_decoding/03_i2c_basic.py` - I2C bus analysis

2. **Automotive Protocols** (1-2 hours)
   - `03_protocol_decoding/04_can_basic.py` - CAN bus basics
   - `03_protocol_decoding/06_lin.py` - LIN protocol
   - `03_protocol_decoding/07_flexray.py` - FlexRay high-speed

3. **Advanced Measurements** (2-3 hours)
   - `04_advanced_analysis/01_jitter_analysis.py` - Timing jitter
   - `04_advanced_analysis/04_eye_diagrams.py` - Eye diagram generation
   - `04_advanced_analysis/06_power_analysis.py` - Power consumption

4. **Signal Processing** (1-2 hours)
   - `02_basic_analysis/03_spectral_basics.py` - FFT and PSD
   - `02_basic_analysis/05_filtering.py` - Signal filtering
   - `04_advanced_analysis/10_correlation.py` - Cross-correlation

**Total**: ~6 hours | **Prerequisites**: Beginner Path | **Next**: Advanced Path

### Advanced Path (8-20 hours)

**For power users and reverse engineers**

1. **Domain-Specific Applications** (2-4 hours)
   - `05_domain_specific/01_automotive_diagnostics.py` - CAN diagnostics
   - `05_domain_specific/03_emc_compliance.py` - EMC testing
   - `05_domain_specific/05_side_channel_basics.py` - Side-channel analysis

2. **Reverse Engineering Techniques** (4-8 hours)
   - `06_reverse_engineering/01_unknown_protocol.py` - Unknown protocol analysis
   - `06_reverse_engineering/02_crc_recovery.py` - CRC algorithm recovery
   - `06_reverse_engineering/03_state_machines.py` - State machine inference
   - `06_reverse_engineering/05_pattern_discovery.py` - Pattern mining
   - `06_reverse_engineering/09_signal_classification.py` - ML classification

3. **Performance Optimization** (2-4 hours)
   - `07_advanced_features/01_lazy_loading.py` - Memory-efficient loading
   - `07_advanced_features/02_memory_management.py` - Large dataset handling
   - `07_advanced_features/03_performance_optimization.py` - Speed optimization

4. **Extensibility** (2-4 hours)
   - `08_extensibility/01_custom_analyzer.py` - Build custom analyzers
   - `08_extensibility/02_custom_plugin.py` - Plugin development
   - `08_extensibility/04_registration.py` - Register custom components

**Total**: ~14 hours | **Prerequisites**: Intermediate Path | **Next**: Expert Path

### Expert Path (20+ hours)

**For framework mastery and production deployment**

1. **Complete Workflows** (6-10 hours)
   - `11_complete_workflows/01_unknown_signal_workflow.py` - End-to-end unknown signal analysis
   - `11_complete_workflows/02_automotive_diagnostics_workflow.py` - Complete automotive workflow
   - `11_complete_workflows/04_production_testing_workflow.py` - Production testing automation

2. **Standards Compliance** (4-6 hours)
   - `12_standards_compliance/01_ieee_181.py` - IEEE 181-2011 pulse measurements
   - `12_standards_compliance/02_ieee_1241.py` - IEEE 1241-2010 ADC testing
   - `12_standards_compliance/04_automotive_standards.py` - Automotive compliance

3. **Integration & Deployment** (4-8 hours)
   - `09_integration/01_ci_cd.py` - Continuous integration
   - `09_integration/02_hardware_integration.py` - Hardware device integration
   - `09_integration/06_web_dashboard.py` - Web dashboard deployment

4. **Advanced Reverse Engineering** (6-10 hours)
   - `06_reverse_engineering/11_bayesian_inference.py` - Bayesian protocol inference
   - `06_reverse_engineering/12_active_learning.py` - Active learning techniques
   - `06_reverse_engineering/14_comprehensive_re.py` - Complete RE workflow
   - `06_reverse_engineering/15_re_tool.py` - Interactive RE tool

**Total**: ~25 hours | **Prerequisites**: Advanced Path

---

## Category Index

### 00_getting_started (3 demos) ⭐ START HERE

Essential first steps for all users.

| Demo | Description | Capabilities | Time | Difficulty |
|------|-------------|--------------|------|------------|
| `00_hello_world.py` | Your first Oscura demo | Load, measure, analyze | 5 min | Beginner |
| `01_core_types.py` | Data structures explained | WaveformTrace, DigitalTrace, metadata | 10 min | Beginner |
| `02_supported_formats.py` | Survey of all loaders and analyzers | All file formats, all protocols | 15 min | Beginner |

**Prerequisites**: Oscura installed
**Next Steps**: 01_data_loading or 02_basic_analysis

### 01_data_loading (10 demos)

Load data from various sources and file formats.

| Demo | Description | File Formats | Time |
|------|-------------|--------------|------|
| `01_oscilloscopes.py` | Oscilloscope file formats | Tektronix WFM, Rigol WFM | 15 min |
| `02_logic_analyzers.py` | Logic analyzer formats | Sigrok, VCD | 15 min |
| `03_automotive_formats.py` | Automotive data formats | CAN trace, DBC, MDF4 | 20 min |
| `04_scientific_formats.py` | Scientific data formats | TDMS, HDF5, WAV, MAT | 20 min |
| `05_custom_binary.py` | Custom binary file loaders | Binary, custom DAQ | 25 min |
| `06_streaming_large_files.py` | Memory-efficient streaming | All formats (large files) | 30 min |
| `07_multi_channel.py` | Multi-channel synchronized data | Multi-channel formats | 20 min |
| `08_network_formats.py` | Network capture formats | PCAP, Touchstone | 20 min |
| `09_lazy_loading.py` | Lazy loading for huge files | All formats (lazy) | 25 min |
| `10_format_conversion.py` | Convert between formats | Format conversion | 20 min |

**Prerequisites**: 00_getting_started
**IEEE Standards**: None
**Related**: 02_basic_analysis, 07_advanced_features

### 02_basic_analysis (8 demos)

Fundamental signal analysis techniques.

| Demo | Description | Measurements | Time |
|------|-------------|--------------|------|
| `01_waveform_basics.py` | Basic waveform measurements | Amplitude, frequency, RMS, Vpp | 15 min |
| `02_digital_basics.py` | Digital signal analysis | Edges, timing, duty cycle | 15 min |
| `03_spectral_basics.py` | Frequency domain analysis | FFT, PSD, THD, harmonics | 20 min |
| `04_measurements.py` | Comprehensive measurements | Rise time, fall time, overshoot | 20 min |
| `05_filtering.py` | Signal filtering techniques | Low-pass, high-pass, band-pass | 20 min |
| `06_triggers.py` | Trigger detection | Edge, level, pattern triggers | 15 min |
| `07_cursors.py` | Cursor measurements | Time cursors, voltage cursors | 15 min |
| `08_statistics.py` | Statistical analysis | Mean, std dev, histograms | 20 min |

**Prerequisites**: 01_data_loading
**IEEE Standards**: IEEE 181-2011 (timing), IEEE 1057-2017 (waveform)
**Related**: 04_advanced_analysis

### 03_protocol_decoding (12 demos)

Decode and analyze communication protocols.

| Demo | Description | Protocol | Bus Type | Time |
|------|-------------|----------|----------|------|
| `01_uart_basic.py` | UART serial communication | UART | Serial | 20 min |
| `02_spi_basic.py` | SPI bus protocol | SPI | Serial | 20 min |
| `03_i2c_basic.py` | I2C bus protocol | I2C | Serial | 20 min |
| `04_can_basic.py` | CAN bus basics | CAN 2.0 | Automotive | 25 min |
| `05_can_fd.py` | CAN-FD protocol | CAN-FD | Automotive | 25 min |
| `06_lin.py` | LIN protocol | LIN | Automotive | 25 min |
| `07_flexray.py` | FlexRay high-speed bus | FlexRay | Automotive | 30 min |
| `08_jtag.py` | JTAG debug interface | JTAG | Debug | 25 min |
| `09_swd.py` | Serial Wire Debug | SWD | Debug | 25 min |
| `10_i2s.py` | I2S audio protocol | I2S | Audio | 20 min |
| `11_usb.py` | USB protocol | USB | Universal | 30 min |
| `12_comprehensive_protocols.py` | Multi-protocol analysis | Multiple | Various | 40 min |

**Prerequisites**: 02_basic_analysis
**IEEE Standards**: None (protocols are vendor/industry standards)
**Related**: 05_domain_specific, 06_reverse_engineering

### 04_advanced_analysis (12 demos)

Advanced signal analysis techniques.

| Demo | Description | Analysis Type | Time |
|------|-------------|---------------|------|
| `01_jitter_analysis.py` | Timing jitter measurement | Jitter | 30 min |
| `02_jitter_decomposition.py` | RJ/DJ/PJ separation | Jitter decomposition | 35 min |
| `03_bathtub_curves.py` | BER bathtub curves | BER analysis | 30 min |
| `04_eye_diagrams.py` | Eye diagram generation | Eye diagrams | 30 min |
| `05_eye_metrics.py` | Eye height, width, Q-factor | Eye metrics | 30 min |
| `06_power_analysis.py` | DC/AC power analysis | Power | 25 min |
| `07_efficiency.py` | Power efficiency calculation | Efficiency | 25 min |
| `08_signal_integrity.py` | S-parameter analysis | Signal integrity | 35 min |
| `09_tdr.py` | Time-domain reflectometry | TDR | 30 min |
| `10_correlation.py` | Cross-correlation analysis | Correlation | 25 min |
| `11_statistics_advanced.py` | Advanced statistical methods | Statistics | 30 min |
| `12_comprehensive_analysis.py` | Multi-analyzer workflow | Comprehensive | 45 min |

**Prerequisites**: 02_basic_analysis, 03_protocol_decoding
**IEEE Standards**: IEEE 181-2011, IEEE 1241-2010, IEEE 1057-2017
**Related**: 05_domain_specific, 12_standards_compliance

### 05_domain_specific (8 demos)

Domain-specific applications and workflows.

| Demo | Description | Domain | Time |
|------|-------------|--------|------|
| `01_automotive_diagnostics.py` | CAN bus diagnostics | Automotive | 35 min |
| `02_automotive_comprehensive.py` | Complete automotive workflow | Automotive | 50 min |
| `03_emc_compliance.py` | EMC testing | EMC | 40 min |
| `04_emc_comprehensive.py` | Complete EMC workflow | EMC | 55 min |
| `05_side_channel_basics.py` | Side-channel analysis intro | Security | 30 min |
| `06_side_channel_dpa.py` | DPA attack analysis | Security | 45 min |
| `07_timing_ieee181.py` | IEEE 181-2011 compliance | Timing | 35 min |
| `08_vintage_logic.py` | Vintage hardware analysis | Retro | 30 min |

**Prerequisites**: 03_protocol_decoding, 04_advanced_analysis
**IEEE Standards**: IEEE 181-2011, MIL-STD-461, CISPR standards
**Related**: 11_complete_workflows, 12_standards_compliance

### 06_reverse_engineering (15 demos)

Protocol reverse engineering and signal analysis.

| Demo | Description | Technique | Time |
|------|-------------|-----------|------|
| `01_unknown_protocol.py` | Unknown protocol analysis | Initial analysis | 30 min |
| `02_crc_recovery.py` | CRC algorithm recovery | CRC inference | 40 min |
| `03_state_machines.py` | State machine inference | State machines | 45 min |
| `04_field_inference.py` | Field boundary detection | Field analysis | 40 min |
| `05_pattern_discovery.py` | Pattern mining | Pattern recognition | 45 min |
| `06_wireshark_export.py` | Wireshark dissector generation | Tool integration | 35 min |
| `07_entropy_analysis.py` | Entropy-based analysis | Entropy | 30 min |
| `08_data_classification.py` | Data type classification | Classification | 35 min |
| `09_signal_classification.py` | ML signal classification | Machine learning | 50 min |
| `10_anomaly_detection.py` | Anomaly detection | Anomaly detection | 40 min |
| `11_bayesian_inference.py` | Bayesian protocol inference | Bayesian methods | 55 min |
| `12_active_learning.py` | Active learning techniques | Active learning | 60 min |
| `13_protocol_dsl.py` | Protocol DSL generation | Code generation | 45 min |
| `14_comprehensive_re.py` | Complete RE workflow | End-to-end RE | 75 min |
| `15_re_tool.py` | Interactive RE tool | Interactive tool | 60 min |

**Prerequisites**: 03_protocol_decoding, 04_advanced_analysis
**IEEE Standards**: None (reverse engineering techniques)
**Related**: 05_domain_specific, 11_complete_workflows

### 07_advanced_features (8 demos)

Advanced API features and performance optimization.

| Demo | Description | Feature | Time |
|------|-------------|---------|------|
| `01_lazy_loading.py` | Lazy loading patterns | Lazy loading | 25 min |
| `02_memory_management.py` | Memory efficiency | Memory management | 30 min |
| `03_performance_optimization.py` | Performance tuning | Optimization | 35 min |
| `04_batch_processing.py` | Batch analysis | Batch processing | 30 min |
| `05_progress_tracking.py` | Progress monitoring | Progress tracking | 20 min |
| `06_cancellation.py` | Cancellable operations | Cancellation | 20 min |
| `07_configuration.py` | Advanced configuration | Configuration | 25 min |
| `08_provenance.py` | Data provenance tracking | Provenance | 30 min |

**Prerequisites**: 02_basic_analysis
**IEEE Standards**: None
**Related**: 01_data_loading, 09_integration

### 08_extensibility (5 demos)

Extend Oscura with custom components.

| Demo | Description | Extension Type | Time |
|------|-------------|----------------|------|
| `01_custom_analyzer.py` | Custom analyzer creation | Analyzer | 40 min |
| `02_custom_plugin.py` | Plugin development | Plugin | 45 min |
| `03_templates.py` | Template usage | Templates | 30 min |
| `04_registration.py` | Component registration | Registration | 35 min |
| `05_plugin_distribution.py` | Plugin packaging | Distribution | 40 min |

**Prerequisites**: 02_basic_analysis, 03_protocol_decoding
**IEEE Standards**: None
**Related**: 09_integration

### 09_integration (6 demos)

Integrate Oscura with external tools and systems.

| Demo | Description | Integration | Time |
|------|-------------|-------------|------|
| `01_ci_cd.py` | CI/CD integration | CI/CD | 30 min |
| `02_hardware_integration.py` | Hardware device integration | Hardware | 40 min |
| `03_external_tools.py` | External tool integration | Tools | 35 min |
| `04_cli_usage.py` | CLI automation | Command-line | 25 min |
| `05_api_usage.py` | Programmatic API usage | API | 30 min |
| `06_web_dashboard.py` | Web dashboard integration | Web | 50 min |

**Prerequisites**: 07_advanced_features, 08_extensibility
**IEEE Standards**: None
**Related**: 10_export_visualization

### 10_export_visualization (6 demos)

Export data and create visualizations.

| Demo | Description | Export Type | Time |
|------|-------------|-------------|------|
| `01_export_formats.py` | All export formats | Multiple formats | 25 min |
| `02_wireshark.py` | Wireshark integration | PCAP, dissectors | 30 min |
| `03_plotting.py` | Plotting techniques | Matplotlib plots | 30 min |
| `04_reporting.py` | Report generation | PDF, HTML reports | 35 min |
| `05_visualization_gallery.py` | Visualization examples | Gallery | 40 min |
| `06_interactive_plots.py` | Interactive plotting | Interactive | 35 min |

**Prerequisites**: 02_basic_analysis
**IEEE Standards**: None
**Related**: 09_integration

### 11_complete_workflows (8 demos)

End-to-end real-world workflows.

| Demo | Description | Application | Time |
|------|-------------|-------------|------|
| `01_unknown_signal_workflow.py` | Complete unknown signal analysis | Reverse engineering | 60 min |
| `02_automotive_diagnostics_workflow.py` | Automotive end-to-end | Automotive | 55 min |
| `03_emc_testing_workflow.py` | EMC testing end-to-end | EMC compliance | 60 min |
| `04_production_testing_workflow.py` | Production testing automation | Manufacturing | 50 min |
| `05_failure_analysis_workflow.py` | Failure analysis workflow | Debugging | 55 min |
| `06_device_characterization_workflow.py` | Device characterization | Characterization | 60 min |
| `07_network_analysis_workflow.py` | Network protocol analysis | Networking | 55 min |
| `08_power_supply_workflow.py` | Power supply analysis | Power electronics | 50 min |

**Prerequisites**: Multiple categories (domain-dependent)
**IEEE Standards**: Various (workflow-dependent)
**Related**: All categories

### 12_standards_compliance (4 demos)

IEEE and industry standards compliance.

| Demo | Description | Standard | Time |
|------|-------------|----------|------|
| `01_ieee_181.py` | IEEE 181-2011 compliance | IEEE 181-2011 (pulse) | 40 min |
| `02_ieee_1241.py` | IEEE 1241-2010 compliance | IEEE 1241-2010 (ADC) | 45 min |
| `03_ieee_1459.py` | IEEE 1459-2010 compliance | IEEE 1459-2010 (power) | 45 min |
| `04_automotive_standards.py` | Automotive standards compliance | SAE, ISO standards | 50 min |

**Prerequisites**: 04_advanced_analysis
**IEEE Standards**: IEEE 181-2011, IEEE 1241-2010, IEEE 1459-2010
**Related**: 05_domain_specific, 11_complete_workflows

---

## Search by Capability

### File Format Loaders

- **Oscilloscopes**: `01_data_loading/01_oscilloscopes.py` (Tektronix, Rigol)
- **Logic Analyzers**: `01_data_loading/02_logic_analyzers.py` (Sigrok, VCD)
- **Automotive**: `01_data_loading/03_automotive_formats.py` (CAN, DBC, MDF4)
- **Scientific**: `01_data_loading/04_scientific_formats.py` (TDMS, HDF5, WAV)
- **Custom Binary**: `01_data_loading/05_custom_binary.py`
- **Network**: `01_data_loading/08_network_formats.py` (PCAP, Touchstone)

### Protocol Decoders

- **UART**: `03_protocol_decoding/01_uart_basic.py`
- **SPI**: `03_protocol_decoding/02_spi_basic.py`
- **I2C**: `03_protocol_decoding/03_i2c_basic.py`
- **CAN**: `03_protocol_decoding/04_can_basic.py`, `05_can_fd.py`
- **LIN**: `03_protocol_decoding/06_lin.py`
- **FlexRay**: `03_protocol_decoding/07_flexray.py`
- **JTAG**: `03_protocol_decoding/08_jtag.py`
- **SWD**: `03_protocol_decoding/09_swd.py`
- **I2S**: `03_protocol_decoding/10_i2s.py`
- **USB**: `03_protocol_decoding/11_usb.py`

### Analysis Capabilities

- **Waveform**: `02_basic_analysis/01_waveform_basics.py`
- **Digital**: `02_basic_analysis/02_digital_basics.py`
- **Spectral**: `02_basic_analysis/03_spectral_basics.py`
- **Jitter**: `04_advanced_analysis/01_jitter_analysis.py`
- **Eye Diagrams**: `04_advanced_analysis/04_eye_diagrams.py`
- **Power**: `04_advanced_analysis/06_power_analysis.py`
- **Signal Integrity**: `04_advanced_analysis/08_signal_integrity.py`
- **TDR**: `04_advanced_analysis/09_tdr.py`

### Reverse Engineering

- **Unknown Protocol**: `06_reverse_engineering/01_unknown_protocol.py`
- **CRC Recovery**: `06_reverse_engineering/02_crc_recovery.py`
- **State Machines**: `06_reverse_engineering/03_state_machines.py`
- **Pattern Discovery**: `06_reverse_engineering/05_pattern_discovery.py`
- **ML Classification**: `06_reverse_engineering/09_signal_classification.py`
- **Wireshark Export**: `06_reverse_engineering/06_wireshark_export.py`

---

## Search by IEEE Standard

### IEEE 181-2011 (Pulse Measurement)
- `02_basic_analysis/04_measurements.py` (rise/fall time)
- `04_advanced_analysis/01_jitter_analysis.py` (timing measurements)
- `05_domain_specific/07_timing_ieee181.py` (full compliance)
- `12_standards_compliance/01_ieee_181.py` (compliance testing)

### IEEE 1241-2010 (ADC Testing)
- `04_advanced_analysis/12_comprehensive_analysis.py` (ADC metrics)
- `12_standards_compliance/02_ieee_1241.py` (compliance testing)

### IEEE 1057-2017 (Waveform Digitization)
- `02_basic_analysis/01_waveform_basics.py` (basic measurements)
- `04_advanced_analysis/11_statistics_advanced.py` (statistical analysis)

### IEEE 1459-2010 (Power Quality)
- `04_advanced_analysis/06_power_analysis.py` (power measurements)
- `12_standards_compliance/03_ieee_1459.py` (compliance testing)

### IEEE 1364-2005 (Verilog/VCD)
- `01_data_loading/02_logic_analyzers.py` (VCD loading)

### IEEE 2414-2010 (Streaming Data)
- `01_data_loading/06_streaming_large_files.py` (streaming)

---

## Troubleshooting

### Demo Won't Run

```bash
# Check Python version (requires 3.12+)
python3 --version

# Verify Oscura installation
python3 -c "import oscura; print(oscura.__version__)"

# Test demo syntax
python3 -m py_compile demos/00_getting_started/00_hello_world.py

# Run with verbose output
python3 demos/00_getting_started/00_hello_world.py --verbose
```

### Import Errors

```bash
# Ensure demos are run from repository root
cd /path/to/oscura
python3 demos/CATEGORY/demo.py

# Or add to PYTHONPATH
export PYTHONPATH=/path/to/oscura:$PYTHONPATH
```

### Validation Failures

```bash
# Skip validation to see demo execute
python3 demos/CATEGORY/demo.py --no-validate

# Check if issue is with demo or with your data
python3 demos/CATEGORY/demo.py --data-file=/your/file
```

---

## Contributing

### Adding a New Demo

1. Choose appropriate category
2. Inherit from BaseDemo
3. Include capabilities metadata
4. Use ValidationSuite
5. Add to this README
6. Test thoroughly

Template:

```python
from demos.common import BaseDemo, ValidationSuite

class MyDemo(BaseDemo):
    name = "My Demo Name"
    description = "Brief description"
    category = "appropriate_category"

    capabilities = [
        "oscura.function1",
        "oscura.function2",
    ]
    ieee_standards = ["IEEE XXX-YYYY"]  # if applicable
    related_demos = ["../other/demo.py"]

    def generate_data(self):
        # Create synthetic test data
        pass

    def run_analysis(self):
        # Perform demonstration
        pass

    def validate_results(self, suite: ValidationSuite):
        suite.check_equal("Test", self.results["value"], expected)
```

---

## Migration Status

**Current State**: Consolidation in progress

**Source**: 3 directories (221 files)
- `demonstrations/` - 161 files (73%)
- `demos/` - 54 files (24%)
- `examples/` - 6 files (3%)

**Target**: Single `demos/` directory (~120 files, 45% reduction)

**Progress**:
- [x] Infrastructure created (common/ utilities)
- [x] Execution plan documented
- [ ] Phase 1: 00_getting_started (0/3)
- [ ] Phase 2: Critical categories (0/30)
- [ ] Phase 3: Advanced categories (0/43)
- [ ] Phase 4: Specialized categories (0/29)
- [ ] Phase 5: Documentation & cleanup

See `.claude/coordination/demos-consolidation-execution-plan.md` for complete roadmap.

---

## Additional Resources

- **Main Documentation**: `/docs/`
- **API Reference**: `/docs/api/`
- **Research Reports**: `/.claude/research-reports/demos-structure-analysis-2026-01-29.md`
- **Migration Plan**: `/.claude/coordination/demos-consolidation-execution-plan.md`

---

**Last Updated**: 2026-01-29
**Maintained By**: Oscura Development Team
**Questions**: See documentation or open an issue
