# Demos Consolidation Task Specification

## Objective
Consolidate 221 demonstration files from 3 directories (`demonstrations/`, `demos/`, `examples/`) into single optimal `demos/` structure with 12 categories and ~120 files (45% reduction).

## Research Context
Complete analysis at `.claude/research-reports/demos-structure-analysis-2026-01-29.md` containing:
- File inventory (all 221 files)
- Capability matrix (49 capabilities)
- Classification (85 ESSENTIAL, 35 VALUABLE, 60 REDUNDANT, 41 OBSOLETE)
- Detailed migration recommendations

## Strategy
**Hybrid Structure**: Combine demonstrations/ excellent organization with demos/ superior ValidationSuite implementation.

### Infrastructure (COMPLETE)
- `demos/common/base_demo.py` - Enhanced BaseDemo with capabilities metadata ✓
- `demos/common/validation.py` - ValidationSuite from demos/ ✓
- `demos/common/formatting.py` - Console formatting ✓
- `demos/common/plotting.py` - Plotting helpers ✓
- `demos/common/__init__.py` - Module exports ✓

## Phase 1: Create Category Structure

Create these 12 category directories:
```
demos/
├── 00_getting_started/      # 3 demos
├── 01_data_loading/          # 10 demos
├── 02_basic_analysis/        # 8 demos
├── 03_protocol_decoding/     # 12 demos
├── 04_advanced_analysis/     # 12 demos
├── 05_domain_specific/       # 8 demos
├── 06_reverse_engineering/   # 15 demos
├── 07_advanced_features/     # 8 demos
├── 08_extensibility/         # 5 demos
├── 09_integration/           # 6 demos
├── 10_export_visualization/  # 6 demos
├── 11_complete_workflows/    # 8 demos
├── 12_standards_compliance/  # 4 demos
└── common/                   # Infrastructure (DONE)
```

## Phase 2: Migrate ESSENTIAL Demos (Priority)

### 00_getting_started (3 demos - HIGHEST PRIORITY)
1. `demonstrations/00_getting_started/00_hello_world.py` → `demos/00_getting_started/00_hello_world.py`
2. `demonstrations/00_getting_started/01_core_types.py` → `demos/00_getting_started/01_core_types.py`
3. `demonstrations/00_getting_started/02_supported_formats.py` → `demos/00_getting_started/02_supported_formats.py`

**Migration Rules**:
- Keep demonstrations/ structure (excellent BaseDemo usage)
- Update imports: `from demonstrations.common` → `from demos.common`
- Verify capabilities, ieee_standards, related_demos metadata present
- Ensure ValidationSuite usage

### 01_data_loading (10 demos - HIGH PRIORITY)
Consolidate demonstrations/01_data_loading (14 files) + parts of demos/02_file_format_io + demos/03_custom_daq:

1. Merge oscilloscope demos → `01_oscilloscopes.py` (Tektronix + Rigol)
2. Merge logic analyzer demos → `02_logic_analyzers.py` (Sigrok + VCD)
3. Merge automotive formats → `03_automotive_formats.py` (CAN/LIN formats)
4. Merge scientific formats → `04_scientific_formats.py` (TDMS/HDF5/WAV)
5. Keep custom binary → `05_custom_binary.py`
6. Keep streaming → `06_streaming_large_files.py`
7. Keep multi-channel → `07_multi_channel.py`
8. Merge network formats → `08_network_formats.py` (PCAP/Touchstone)
9. Keep lazy loading → `09_lazy_loading.py`
10. Add format conversion → `10_format_conversion.py`

### 02_basic_analysis (8 demos)
Merge demonstrations/02_basic_analysis + demos/01_waveform_analysis:

1. Merge waveform basics → `01_waveform_basics.py` (amplitude, frequency, RMS)
2. `02_digital_basics.py` (edge detection, timing)
3. Merge spectral → `03_spectral_basics.py` (FFT, PSD, THD)
4. `04_measurements.py`
5. `05_filtering.py`
6. `06_triggers.py`
7. `07_cursors.py`
8. `08_statistics.py`

### 03_protocol_decoding (12 demos)
Merge demonstrations/03_protocol_decoding + demos/04_serial_protocols + demos/05_protocol_decoding:

1. Merge UART demos (4 sources) → `01_uart_basic.py`
2. Merge SPI demos (4 sources) → `02_spi_basic.py`
3. Merge I2C demos (4 sources) → `03_i2c_basic.py`
4. Merge CAN demos (5 sources) → `04_can_basic.py`
5. `05_can_fd.py`
6. Merge LIN (5 sources) → `06_lin.py`
7. Merge FlexRay (4 sources) → `07_flexray.py`
8. `08_jtag.py`
9. `09_swd.py`
10. `10_i2s.py`
11. `11_usb.py`
12. `12_comprehensive_protocols.py` (multi-protocol)

### 04_advanced_analysis (12 demos)
Merge demonstrations/04_advanced_analysis + demos/13_jitter_analysis + demos/14_power_analysis + demos/15_signal_integrity:

1-3. Jitter analysis (merge multiple)
4-5. Eye diagrams
6-7. Power analysis
8-9. Signal integrity
10-12. Statistics and correlation

### 05_domain_specific (8 demos)
Merge demonstrations/05_domain_specific + demos/08_automotive_protocols + demos/09_automotive + demos/16_emc_compliance:

1-2. Automotive (diagnostics, comprehensive)
3-4. EMC compliance
5-6. Side-channel analysis
7. IEEE 181 timing
8. Vintage logic

### 06_reverse_engineering (15 demos)
Merge demonstrations/06_reverse_engineering + demos/07_protocol_inference + demos/17_signal_reverse_engineering + demos/18_advanced_inference + examples/ml_signal_classification_demo.py + examples/export/wireshark_dissector_demo.py:

1. Unknown protocol
2-5. CRC recovery, state machines, field inference, pattern discovery
6. Wireshark export (from examples/)
7-10. Entropy, data classification, signal classification (from examples/), anomaly detection
11-12. Bayesian, active learning
13-15. Protocol DSL, comprehensive RE, RE tool

### 07_advanced_features (8 demos)
From demonstrations/07_advanced_api + demonstrations/09_batch_processing:

1-8. Lazy loading, memory, performance, batch, progress, cancellation, config, provenance

### 08_extensibility (5 demos)
From demonstrations/08_extensibility:

1-5. Custom analyzer, plugin, templates, registration, distribution

### 09_integration (6 demos)
From demonstrations/11_integration + examples/web_dashboard_example.py:

1-6. CI/CD, hardware, external tools, CLI, API, web dashboard (from examples/)

### 10_export_visualization (6 demos)
From demonstrations/15_export_visualization:

1-6. Export formats, Wireshark, plotting, reporting, gallery, interactive

### 11_complete_workflows (8 demos)
Merge demonstrations/16_complete_workflows + demos/19_complete_workflows:

1-8. Unknown signal, automotive diagnostics, EMC testing, production testing, failure analysis, device characterization, network analysis, power supply

### 12_standards_compliance (4 demos)
From demonstrations/19_standards_compliance:

1-4. IEEE 181, IEEE 1241, IEEE 1459, Automotive standards

## Migration Checklist (Per Demo)
- [ ] Read source demo file
- [ ] Update imports: `from demonstrations.common` → `from demos.common` or `from demos.common` (keep as is)
- [ ] Verify class inherits from BaseDemo
- [ ] Ensure capabilities metadata present
- [ ] Ensure ieee_standards metadata present (if applicable)
- [ ] Update related_demos paths
- [ ] Verify ValidationSuite usage in validate_results()
- [ ] Update any hardcoded paths
- [ ] Write to new location
- [ ] Test syntax (at minimum)

## Merge Strategy (For Redundant Demos)
When merging multiple demos into one:

1. **Choose base**: Usually demonstrations/ version (better structure)
2. **Extract value**: Pull ValidationSuite checks from demos/ version
3. **Merge metadata**: Combine capabilities, standards, related demos
4. **Consolidate data**: Use best synthetic data generator
5. **Add sections**: Use subsection headers for different aspects
6. **Update validation**: Comprehensive checks for all merged content

Example merge comment at top:
```python
"""UART Protocol Decoding - Basic Demonstration.

Consolidated from:
- demonstrations/03_protocol_decoding/01_uart_basic.py (base structure)
- demos/04_serial_protocols/uart_demo.py (ValidationSuite enhancements)
- demonstrations/03_protocol_decoding/02_uart_error_handling.py (error cases)
- demos/04_serial_protocols/uart_advanced.py (advanced features)

Demonstrates all basic UART decoding capabilities in one comprehensive demo.
"""
```

## Output Artifacts
After migration:
- New `demos/` directory with 12 categories
- ~120 demonstration files (down from 221)
- 100% BaseDemo adoption
- 100% ValidationSuite coverage
- All capabilities metadata preserved
- All IEEE standards references preserved

## DO NOT (Critical Constraints)
- DO NOT delete old directories yet (archive them in Phase 5)
- DO NOT modify demos/common/ (already correct)
- DO NOT skip capabilities/standards metadata
- DO NOT create demos without ValidationSuite
- DO NOT exceed target file counts per category significantly
- DO NOT mix in examples/ content without review (most is obsolete)

## Next Steps After This Task
1. Technical writer to create README.md catalog
2. Update CLAUDE.md PROJECT LAYOUT section
3. Update main README.md examples section
4. Archive old directories
5. Update CHANGELOG.md
6. Run validators

## Success Criteria
- All 12 category directories created
- 85+ ESSENTIAL demos migrated successfully
- All demos follow BaseDemo pattern
- All demos have ValidationSuite
- Capabilities metadata preserved
- No broken imports
- File count within targets (~120 total)
