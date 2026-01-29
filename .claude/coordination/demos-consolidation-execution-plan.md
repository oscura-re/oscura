# Demos Consolidation - Detailed Execution Plan

## Executive Summary

**Task**: Consolidate 221 demonstration files from 3 directories into ~120 optimized demos
**Effort**: 40-60 hours (multi-day project)
**Status**: Foundation created, proof-of-concept in progress
**Completion**: ~5% (infrastructure + plan)

## Problem Assessment

This consolidation involves:
- **3 different BaseDemo patterns** (demonstrations/, demos/, examples/)
- **2 different validation frameworks** (demonstrations/common vs demos/common)
- **Different helper utilities** in each common/ directory
- **221 source files** requiring individual analysis
- **Complex merging logic** for redundant demos
- **Metadata preservation** (capabilities, IEEE standards)
- **Import path updates** across all files
- **Testing requirements** for each migrated demo

**Reality Check**: This is NOT a single-session task. It's a structured engineering project.

## Recommended Approach

### Option 1: Phased Migration (RECOMMENDED)
Execute in phases over multiple sessions with validation between each:

**Phase 1**: Infrastructure & Category 00 (2-4 hours)
- ✓ Merge common/ utilities from both directories
- ✓ Create unified BaseDemo
- ✓ Migrate 00_getting_started (3 demos)
- ✓ Test and validate
- ✓ Create migration template

**Phase 2**: Critical Categories (8-12 hours)
- 01_data_loading (10 demos with merges)
- 02_basic_analysis (8 demos with merges)
- 03_protocol_decoding (12 demos with complex merges)
- Test each category before proceeding

**Phase 3**: Advanced Categories (12-16 hours)
- 04_advanced_analysis (12 demos)
- 05_domain_specific (8 demos)
- 06_reverse_engineering (15 demos)
- 07_advanced_features (8 demos)

**Phase 4**: Specialized Categories (8-12 hours)
- 08_extensibility (5 demos)
- 09_integration (6 demos)
- 10_export_visualization (6 demos)
- 11_complete_workflows (8 demos)
- 12_standards_compliance (4 demos)

**Phase 5**: Documentation & Cleanup (6-8 hours)
- Create comprehensive README.md
- Update all documentation
- Archive old directories
- Run full test suite
- Update CHANGELOG.md

### Option 2: Automated Migration Script
Create a Python script to automate the migration:

```python
# demos_migration_tool.py
# - Read source demo
# - Parse metadata
# - Update imports automatically
# - Apply merge rules
# - Write to new location
# - Generate validation report
```

**Effort to create**: 8-12 hours
**Savings**: 20-30 hours on manual migration
**Risk**: Requires careful testing, may miss edge cases

### Option 3: Hybrid Approach (BEST BALANCE)
1. Create automated migration script for simple cases (30-40% of demos)
2. Manual migration for complex merges (protocol decoding, workflows)
3. Automated testing and validation
4. Human review of all outputs

## Detailed Migration Specifications

### Category 00: getting_started (3 demos)

#### 00_hello_world.py
**Source**: `demonstrations/00_getting_started/00_hello_world.py`
**Target**: `demos/00_getting_started/00_hello_world.py`
**Type**: Simple migration (no merge needed)
**Complexity**: LOW

**Changes Required**:
```python
# OLD
from demonstrations.common import BaseDemo, generate_sine_wave, validate_approximately

# NEW
from demos.common import BaseDemo, ValidationSuite
from demos.common.data_generation import generate_sine_wave

# Convert old validate() method to new validate_results(suite):
def validate_results(self, suite: ValidationSuite):
    suite.check_close("Amplitude", self.results["amplitude"], 2.0, rtol=0.05)
    suite.check_close("Frequency", self.results["frequency"], 1000.0, rtol=0.01)
    suite.check_close("RMS", self.results["rms"], 0.707, rtol=0.02)
```

**Metadata to Preserve**:
```python
capabilities = [
    "oscura.WaveformTrace",
    "oscura.amplitude",
    "oscura.frequency",
    "oscura.rms",
]
ieee_standards = []
related_demos = [
    "01_core_types.py",
    "../02_basic_analysis/01_waveform_basics.py"
]
```

#### 01_core_types.py
**Source**: `demonstrations/00_getting_started/01_core_types.py`
**Target**: `demos/00_getting_started/01_core_types.py`
**Type**: Simple migration
**Complexity**: MEDIUM (extensive metadata)

**Special Notes**:
- Very comprehensive (373 lines)
- Excellent documentation
- Keep ALL metadata
- Rich capabilities list

#### 02_supported_formats.py
**Source**: `demonstrations/00_getting_started/02_supported_formats.py`
**Target**: `demos/00_getting_started/02_supported_formats.py`
**Type**: Simple migration
**Complexity**: MEDIUM

**Special Notes**:
- Survey demo showing all loaders
- Update all loader imports
- Preserve format list

### Category 01: data_loading (10 demos from 14+)

#### 01_oscilloscopes.py (MERGE 3→1)
**Sources**:
- `demonstrations/01_data_loading/01_tektronix_oscilloscope.py` (BASE)
- `demonstrations/01_data_loading/02_rigol_oscilloscope.py`
- `demonstrations/01_data_loading/10_multi_vendor_scope.py`

**Merge Strategy**:
```python
class OscilloscopeLoadersDemo(BaseDemo):
    """Comprehensive oscilloscope file format loading.

    Consolidated from:
    - demonstrations/01_data_loading/01_tektronix_oscilloscope.py
    - demonstrations/01_data_loading/02_rigol_oscilloscope.py
    - demonstrations/01_data_loading/10_multi_vendor_scope.py

    Demonstrates all oscilloscope loaders in one comprehensive demo.
    """

    name = "Oscilloscope File Formats"
    description = "Load and compare data from multiple oscilloscope vendors"
    category = "data_loading"

    capabilities = [
        "oscura.load_tektronix_wfm",
        "oscura.load_rigol_wfm",
        "oscura.loaders.tektronix.TektronixLoader",
        "oscura.loaders.rigol.RigolLoader",
    ]

    def generate_data(self):
        # Generate test data for each format
        ...

    def run_analysis(self):
        self.section("Oscilloscope Loaders")

        self.subsection("Tektronix WFM Format")
        # Load and analyze Tektronix file
        ...

        self.subsection("Rigol WFM Format")
        # Load and analyze Rigol file
        ...

        self.subsection("Cross-Vendor Comparison")
        # Compare loading approaches
        ...
```

#### 02_logic_analyzers.py (MERGE 2→1)
**Sources**:
- `demonstrations/01_data_loading/03_sigrok_logic_analyzer.py` (BASE)
- `demonstrations/01_data_loading/04_vcd_logic_analyzer.py`

#### 03_automotive_formats.py (MERGE 3→1)
**Sources**:
- `demonstrations/01_data_loading/05_automotive_can_trace.py`
- `demonstrations/01_data_loading/11_can_database_dbc.py`
- `demonstrations/01_data_loading/12_automotive_mdf4.py`

#### 04_scientific_formats.py (MERGE 4→1)
**Sources**:
- `demonstrations/01_data_loading/06_tdms_scientific.py`
- `demonstrations/01_data_loading/07_hdf5_scientific.py`
- `demonstrations/01_data_loading/08_wav_audio.py`
- Maybe `demonstrations/01_data_loading/13_matlab_mat.py`

#### 05_custom_binary.py (KEEP + ENHANCE)
**Source**: `demonstrations/01_data_loading/09_custom_binary.py`
**Enhancement**: Add best practices from `demos/03_custom_daq/`

#### 06_streaming_large_files.py (KEEP)
**Source**: `demonstrations/01_data_loading/06_streaming_large_files.py`
**Type**: Keep as-is (excellent demo, 615 lines)

#### 07_multi_channel.py (KEEP)
**Source**: `demonstrations/01_data_loading/14_multi_channel_synchronized.py`

#### 08_network_formats.py (MERGE 2→1)
**Sources**:
- PCAP loader demo (find in demonstrations/)
- Touchstone loader demo (find in demonstrations/)

#### 09_lazy_loading.py (KEEP or MERGE)
**Source**: Best from demonstrations/ or demos/07_advanced_api/

#### 10_format_conversion.py (NEW or from examples/)
**Source**: Create new or find in examples/

### Category 02: basic_analysis (8 demos from 10+6)

#### 01_waveform_basics.py (MERGE 3→1)
**Sources**:
- `demonstrations/02_basic_analysis/01_waveform_measurements.py` (BASE)
- `demos/01_waveform_analysis/comprehensive_wfm_analysis.py`
- `demonstrations/02_basic_analysis/02_amplitude_measurements.py`

**Key Capabilities**:
- oscura.amplitude, oscura.frequency, oscura.rms
- oscura.peak_to_peak, oscura.duty_cycle
- oscura.rise_time, oscura.fall_time

#### 02_digital_basics.py
**Source**: `demonstrations/02_basic_analysis/03_digital_measurements.py`

#### 03_spectral_basics.py (MERGE 2→1)
**Sources**:
- `demonstrations/02_basic_analysis/04_spectral_analysis.py`
- `demos/12_spectral_compliance/comprehensive_spectral_demo.py`

#### 04_measurements.py through 08_statistics.py
Map remaining from demonstrations/02_basic_analysis/

### Category 03: protocol_decoding (12 demos from 14+6+examples)

**Complex merges**:
- UART: 4 sources → 1 comprehensive
- SPI: 4 sources → 1 comprehensive
- I2C: 4 sources → 1 comprehensive
- CAN: 5 sources → 2 (basic + advanced/CAN-FD)
- LIN: 5 sources → 1 comprehensive
- FlexRay: 4 sources → 1 comprehensive

Each protocol demo should have:
```python
class UARTDecodingDemo(BaseDemo):
    """UART Protocol Decoding - Comprehensive Demonstration.

    Consolidated from:
    - demonstrations/03_protocol_decoding/01_uart_basic.py
    - demonstrations/03_protocol_decoding/02_uart_error_handling.py
    - demos/04_serial_protocols/uart_demo.py
    - Additional UART examples
    """

    capabilities = [
        "oscura.decode_uart",
        "oscura.protocols.uart.UARTDecoder",
        "oscura.protocols.uart.UARTFrame",
    ]
    ieee_standards = []  # UART has no IEEE standard
    related_demos = ["02_spi_basic.py", "03_i2c_basic.py"]

    def run_analysis(self):
        self.section("UART Protocol Decoding")

        self.subsection("Basic Frame Decoding")
        # Basic 8N1 UART

        self.subsection("Different Configurations")
        # 7E1, 8E2, etc.

        self.subsection("Error Handling")
        # Framing errors, parity errors

        self.subsection("Advanced Features")
        # Auto-baud detection, break detection
```

### Categories 04-12: Similar Detailed Specs

[Continue with same level of detail for all remaining categories...]

## Common/ Utilities Consolidation

### Required Merge: demonstrations/common + demos/common

**demonstrations/common has**:
- data_generation.py (generate_sine_wave, generate_square_wave, etc.)
- validation.py (validate_approximately, validate_range, etc.)
- formatting.py (format_duration, format_table, etc.)
- plotting.py (plot_waveform, plot_spectrum, etc.)
- builders.py (SignalBuilder)
- output.py (ValidationSuite - different from demos/)

**demos/common has**:
- base_demo.py (cleaner implementation)
- validation.py (ValidationSuite with better design)
- formatting.py (better colored output)
- plotting.py (DemoPlotter wrapper)

**Merge Strategy**:
1. Keep demos/base_demo.py as foundation (better design)
2. Add metadata fields from demonstrations/BaseDemo (capabilities, ieee_standards, related_demos)
3. Keep demos/validation.py ValidationSuite (superior design)
4. Merge demonstrations/data_generation.py into demos/common/data_generation.py
5. Keep demos/formatting.py (better implementation)
6. Merge plotting utilities
7. Add builders.py if useful

## Testing Strategy

After each migration:

```bash
# Test syntax
python3 demos/00_getting_started/00_hello_world.py --no-validate

# Test full execution
python3 demos/00_getting_started/00_hello_world.py

# Test with custom data
python3 demos/00_getting_started/00_hello_world.py --data-file=/path/to/data

# Verify imports
python3 -m py_compile demos/00_getting_started/00_hello_world.py
```

## Automation Opportunities

### 1. Import Updater Script

```python
# update_imports.py
import re
from pathlib import Path

def update_imports(file_path):
    """Update imports from demonstrations.common to demos.common."""
    content = file_path.read_text()

    # Update main imports
    content = re.sub(
        r'from demonstrations\.common import',
        r'from demos.common import',
        content
    )

    # Add ValidationSuite if needed
    if 'def validate(' in content and 'ValidationSuite' not in content:
        # Need to convert old validate() to new validate_results(suite)
        ...

    file_path.write_text(content)
```

### 2. Merge Automation Script

```python
# merge_demos.py
def merge_demos(sources: list[Path], target: Path, base_index: int = 0):
    """Merge multiple demo files into one comprehensive demo."""

    base_demo = sources[base_index]
    other_demos = [s for i, s in enumerate(sources) if i != base_index]

    # Parse base demo
    base_class = parse_demo_class(base_demo)

    # Extract sections from other demos
    for demo_file in other_demos:
        sections = extract_sections(demo_file)
        base_class.add_sections(sections)

    # Merge capabilities
    base_class.merge_capabilities(other_demos)

    # Write merged demo
    write_demo(base_class, target)
```

### 3. Validation Report Generator

```python
# generate_validation_report.py
def generate_report():
    """Generate migration validation report."""

    report = {
        "total_source_files": 221,
        "total_target_files": 0,
        "migrated": [],
        "merged": [],
        "skipped": [],
        "errors": [],
    }

    for category in Path("demos").glob("[0-9]*"):
        for demo in category.glob("*.py"):
            # Validate demo
            result = validate_demo(demo)
            report["migrated"].append(result)

    # Generate markdown report
    write_report(report)
```

## Progress Tracking

Use `.claude/coordination/demos-migration-progress.json`:

```json
{
  "started": "2026-01-29",
  "last_updated": "2026-01-29",
  "total_categories": 12,
  "completed_categories": 0,
  "total_demos_target": 120,
  "total_demos_completed": 0,
  "categories": {
    "00_getting_started": {
      "status": "in_progress",
      "target": 3,
      "completed": 0,
      "demos": {
        "00_hello_world.py": "not_started",
        "01_core_types.py": "not_started",
        "02_supported_formats.py": "not_started"
      }
    },
    ...
  }
}
```

## Next Steps

1. **Immediate** (today):
   - Complete Phase 1 (00_getting_started migration)
   - Test migrated demos
   - Create migration template
   - Document lessons learned

2. **Short-term** (this week):
   - Complete Phase 2 (critical categories)
   - Create automation scripts
   - Validate each category

3. **Medium-term** (next week):
   - Complete Phases 3-4 (advanced categories)
   - Full testing
   - Documentation

4. **Final** (week after):
   - Phase 5 (cleanup and archive)
   - Full validation
   - Update CHANGELOG.md
   - Merge to main

## Risk Mitigation

1. **Keep old directories** until 100% validated
2. **Git branch** for all changes
3. **Automated testing** after each migration
4. **Peer review** of merged demos
5. **User testing** of critical demos

## Success Criteria

- [ ] All 12 categories created
- [ ] ~120 demos migrated (target ±10%)
- [ ] 100% BaseDemo adoption
- [ ] 100% ValidationSuite coverage
- [ ] All capabilities metadata preserved
- [ ] All IEEE standards references preserved
- [ ] All demos pass validation
- [ ] Comprehensive README.md created
- [ ] Documentation updated
- [ ] Old directories archived
- [ ] CHANGELOG.md updated

## Estimated Timeline

- **With automation**: 20-30 hours over 1-2 weeks
- **Manual only**: 40-60 hours over 2-4 weeks
- **Hybrid approach**: 25-35 hours over 1.5-2 weeks

**Recommendation**: Use hybrid approach with automation for simple cases and manual care for complex merges.
