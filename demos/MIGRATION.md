# Demos Consolidation Migration

## Overview

This document tracks the migration from dual demonstration directories (`demos/` and `demonstrations/`) to a single, unified `demos/` structure.

**Status**: 🟡 In Progress
**Started**: 2026-01-29
**Target Completion**: TBD

## Migration Goals

1. **Consolidate** two parallel demonstration systems into one authoritative location
2. **Preserve** best features from both systems (ValidationSuite from `demos/`, metadata from `demonstrations/`)
3. **Reorganize** by learning progression (00-12 categories)
4. **Enhance** common infrastructure with capabilities metadata
5. **Standardize** all demos on enhanced BaseDemo pattern

## Directory Structure

### New Organization (Target)

```
demos/
├── 00_getting_started/     ✅ COMPLETE (3 demos)
├── 01_data_loading/        🚧 PLANNED (10 demos)
├── 02_basic_analysis/      🚧 PLANNED (8 demos)
├── 03_protocol_decoding/   🚧 PLANNED (8 demos)
├── 04_advanced_analysis/   🚧 PLANNED (7 demos)
├── 05_domain_specific/     🚧 PLANNED (6 demos)
├── 06_reverse_engineering/ 🚧 PLANNED (8 demos)
├── 07_advanced_features/   🚧 PLANNED (5 demos)
├── 08_extensibility/       🚧 PLANNED (4 demos)
├── 09_integration/         🚧 PLANNED (4 demos)
├── 10_export_visualization/🚧 PLANNED (3 demos)
├── 11_complete_workflows/  🚧 PLANNED (5 demos)
├── 12_standards_compliance/🚧 PLANNED (4 demos)
└── common/                 ✅ COMPLETE
    ├── __init__.py
    ├── base_demo.py        (enhanced with metadata)
    ├── validation.py
    ├── validation_helpers.py
    ├── formatting.py
    ├── plotting.py
    └── data_generation.py  (added)
```

## Completed Work

### Infrastructure ✅

**Status**: Complete
**Files Created**: 7
**Date**: 2026-01-29

- [x] Created unified `demos/common/` infrastructure
- [x] Enhanced `BaseDemo` with capabilities/IEEE standards/related_demos attributes
- [x] Created `data_generation.py` with signal generation helpers
- [x] Created `validation_helpers.py` for simple validation functions
- [x] All existing common files preserved (validation.py, formatting.py, plotting.py)

**Key Enhancement**: BaseDemo now supports:
```python
class MyDemo(BaseDemo):
    capabilities = ["oscura.feature_x", "oscura.feature_y"]
    ieee_standards = ["IEEE 181-2011"]
    related_demos = ["path/to/related.py"]
```

### Category 00: Getting Started ✅

**Status**: Complete
**Demos**: 3/3
**Date**: 2026-01-29

- [x] `00_hello_world.py` - Migrated from demonstrations/
  - Uses new BaseDemo with ValidationSuite
  - Includes capabilities metadata
  - Self-contained data generation
  - Comprehensive validation

- [x] `01_core_types.py` - Migrated and streamlined
  - Demonstrates all core Oscura data types
  - Uses new infrastructure
  - Validates type creation

- [x] `02_supported_formats.py` - Migrated and simplified
  - Overview of 21+ file formats
  - Categorized by equipment type
  - Shows auto-detection capabilities

- [x] Created `00_getting_started/README.md`
- [x] Created `00_getting_started/__init__.py`
- [x] All demos syntax-validated

## Planned Work

### Category 01: Data Loading (10 demos) 🚧

**Target Files**:
1. `01_oscilloscopes.py` - Merge Tektronix/Rigol loaders
2. `02_logic_analyzers.py` - Merge Sigrok/VCD loaders
3. `03_automotive_formats.py` - CAN (BLF/ASC), MF4
4. `04_scientific_formats.py` - TDMS, HDF5, WAV
5. `05_custom_binary.py` - Custom format loader example
6. `06_streaming_large_files.py` - Memory-efficient loading
7. `07_multi_channel.py` - Multi-channel handling
8. `08_network_formats.py` - PCAP/PCAPNG
9. `09_lazy_loading.py` - Lazy loading patterns
10. `10_format_conversion.py` - Format conversion utilities

**Source Material**:
- `demonstrations/01_data_loading/` (comprehensive examples)
- `demos/02_file_format_io/` (VCD loader)
- `demos/03_custom_daq/` (custom loaders)

### Category 02: Basic Analysis (8 demos) 🚧

**Target Files**:
1. `01_waveform_basics.py` - Amplitude, frequency, duty cycle
2. `02_digital_basics.py` - Digital signal analysis
3. `03_spectral_basics.py` - FFT, PSD basics
4. `04_measurements.py` - Comprehensive measurements
5. `05_filtering.py` - Filter design and application
6. `06_triggers.py` - Trigger detection
7. `07_cursors.py` - Cursor measurements
8. `08_statistics.py` - Statistical analysis

**Source Material**:
- `demonstrations/02_basic_analysis/`
- `demos/01_waveform_analysis/comprehensive_wfm_analysis.py`
- `demos/12_spectral_compliance/`

### Remaining Categories (03-12)

See "Directory Structure" above for complete list. Each will follow the same migration pattern:
1. Identify source demos
2. Merge/consolidate similar demos
3. Update to new BaseDemo pattern
4. Add comprehensive validation
5. Create category README

## Migration Process

### Per-Demo Checklist

For each demo being migrated:

- [ ] Read source demo(s) from both directories
- [ ] Identify best implementation approach
- [ ] Create new demo file in target location
- [ ] Update imports to use `demos.common`
- [ ] Convert to new BaseDemo pattern:
  - [ ] Add capabilities metadata
  - [ ] Add IEEE standards (if applicable)
  - [ ] Add related_demos references
  - [ ] Use ValidationSuite for validation
  - [ ] Use data_generation helpers for synthetic data
- [ ] Test syntax: `python -m py_compile <file>`
- [ ] Test execution: `python <file>`
- [ ] Document in category README

### Per-Category Checklist

For each category:

- [ ] Create category directory
- [ ] Migrate all demos
- [ ] Create `__init__.py`
- [ ] Create `README.md` with:
  - [ ] Category overview
  - [ ] Individual demo descriptions
  - [ ] Run instructions
  - [ ] Next steps
- [ ] Update `demos/README.md` with status

## Source Material Mapping

### From `demonstrations/`

| Source Category | Target Category | Notes |
|-----------------|-----------------|-------|
| 00_getting_started | 00_getting_started | ✅ Complete |
| 01_data_loading | 01_data_loading | Best structure, use as base |
| 02_basic_analysis | 02_basic_analysis | Merge with demos/01_waveform_analysis |
| 03_protocol_decoding | 03_protocol_decoding | Merge with demos/04_serial_protocols |
| 04_advanced_analysis | 04_advanced_analysis | Standards-focused, preserve |
| 05_domain_specific | 05_domain_specific | Automotive, power, RF |
| 06_reverse_engineering | 06_reverse_engineering | Merge with demos/17_signal_reverse_engineering |
| 07_advanced_api | 07_advanced_features | Lazy loading, batch processing |
| 08_extensibility | 08_extensibility | Custom loaders/decoders |
| 09_batch_processing | 07_advanced_features | Merge into advanced_features |
| 11_integration | 09_integration | Tool integration examples |
| 15_export_visualization | 10_export_visualization | Export formats |
| 16_complete_workflows | 11_complete_workflows | End-to-end pipelines |
| 19_standards_compliance | 12_standards_compliance | IEEE standards demos |

### From `demos/`

| Source Category | Target Category | Notes |
|-----------------|-----------------|-------|
| 01_waveform_analysis | 02_basic_analysis | Merge with demonstrations/02 |
| 02_file_format_io | 01_data_loading | VCD loader example |
| 03_custom_daq | 01_data_loading | Custom format examples |
| 04_serial_protocols | 03_protocol_decoding | Merge with demonstrations/03 |
| 05_protocol_decoding | 03_protocol_decoding | Comprehensive protocol demo |
| 06_udp_packet_analysis | 03_protocol_decoding | Network protocols |
| 07_protocol_inference | 06_reverse_engineering | CRC, state machines |
| 08_automotive_protocols | 05_domain_specific | FlexRay, LIN |
| 09_automotive | 05_domain_specific | OBD-II, UDS, J1939 |
| 10_timing_measurements | 04_advanced_analysis | IEEE 181 pulse |
| 11_mixed_signal | 04_advanced_analysis | Mixed signal analysis |
| 12_spectral_compliance | 04_advanced_analysis | IEEE 1241 ADC |
| 13_jitter_analysis | 04_advanced_analysis | IEEE 2414 jitter |
| 14_power_analysis | 05_domain_specific | Power quality |
| 15_signal_integrity | 04_advanced_analysis | TDR, S-params |
| 16_emc_compliance | 12_standards_compliance | CISPR/IEC EMC |
| 17_signal_reverse_engineering | 06_reverse_engineering | Unknown signal workflows |
| 18_advanced_inference | 06_reverse_engineering | ML-based inference |
| 19_complete_workflows | 11_complete_workflows | Production pipelines |

## Timeline Estimate

### Phase 1: Foundation (Complete) ✅
- Infrastructure setup
- Category 00 migration
- **Effort**: 2-3 hours
- **Status**: Done (2026-01-29)

### Phase 2: Core Categories (Target: Week 1)
- Category 01: Data Loading (10 demos)
- Category 02: Basic Analysis (8 demos)
- Category 03: Protocol Decoding (8 demos)
- **Effort**: 15-20 hours
- **Status**: Pending

### Phase 3: Advanced Categories (Target: Week 2)
- Category 04: Advanced Analysis (7 demos)
- Category 05: Domain Specific (6 demos)
- Category 06: Reverse Engineering (8 demos)
- **Effort**: 15-18 hours
- **Status**: Pending

### Phase 4: Final Categories (Target: Week 3)
- Category 07-12 (remaining categories)
- Final validation and testing
- Documentation updates
- **Effort**: 12-15 hours
- **Status**: Pending

**Total Estimated Effort**: 45-55 hours

## Testing Strategy

### Continuous Testing

After each demo migration:
1. Syntax check: `python -m py_compile <demo>.py`
2. Execution test: `python <demo>.py`
3. Validation test: `python <demo>.py` (check ValidationSuite passes)

### Category Testing

After each category completion:
1. Run all demos in category
2. Check all validations pass
3. Verify README accuracy
4. Test example commands

### Final Testing

Before marking migration complete:
1. Run `demos/validate_all_demos.py` (if it exists)
2. Test random sampling of demos across categories
3. Verify all README links work
4. Check CHANGELOG.md is updated

## Post-Migration Tasks

After migration completes:

- [ ] Update main `README.md` to reference new structure
- [ ] Deprecate old `demonstrations/` directory (or remove)
- [ ] Update `CONTRIBUTING.md` with new demo guidelines
- [ ] Update `.claude/paths.yaml` if needed
- [ ] Create PR with migration changes
- [ ] Update CI/CD to test new demo structure
- [ ] Announce migration completion

## Notes

### Design Decisions

**Why consolidate?**
- Reduces confusion (single source of truth)
- Easier to maintain
- Better discoverability
- Consistent quality

**Why keep `demos/common/` infrastructure?**
- More comprehensive than `demonstrations/common/`
- Better ValidationSuite
- Better formatting utilities
- More mature plotting support

**Why add metadata to BaseDemo?**
- From `demonstrations/` approach
- Better capability tracking
- Easier to generate indexes
- Supports related demo navigation

### Lessons Learned

1. **Infrastructure first**: Getting common code right saves time later
2. **Incremental migration**: Category-by-category is manageable
3. **Preserve best features**: Both systems had strengths to merge
4. **Validation crucial**: Self-validating demos prevent regressions

## Questions/Issues

Track any migration questions or issues here:

1. **Q**: Should we keep old directories during migration?
   **A**: Yes, until migration complete and tested

2. **Q**: How to handle demos with external data dependencies?
   **A**: Generate synthetic equivalent data in-demo

3. **Q**: What to do with demos that don't fit new structure?
   **A**: Evaluate case-by-case, possibly create "special" category

---

**Last Updated**: 2026-01-29
**Next Update**: After Category 01 completion
