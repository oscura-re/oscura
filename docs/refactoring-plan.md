# Comprehensive Refactoring Plan: Long Functions

## Executive Summary

- **Total Functions**: 296 functions >100 lines or complexity >15
- **Affected Files**: 197 files
- **Estimated Effort**: 74 hours (4,440 minutes @ 15min/function)
- **Current Status**: Analysis complete, ready for systematic refactoring

## Top 20 Priority Functions (by Complexity)

| Lines | Complexity | Location | Function |
|-------|------------|----------|----------|
| 150 | 41 | src/oscura/loaders/csv_loader.py:296 | `_load_basic` |
| 302 | 38 | src/oscura/workflows/complete_re.py:124 | `full_protocol_re` |
| 210 | 35 | src/oscura/guidance/wizard.py:184 | `run` |
| 275 | 33 | src/oscura/visualization/power.py:18 | `plot_power_profile` |
| 230 | 33 | src/oscura/inference/signal_intelligence.py:38 | `classify_signal` |
| 116 | 33 | src/oscura/iot/lorawan/mac_commands.py:53 | `parse_mac_command` |
| 104 | 31 | src/oscura/reporting/output.py:18 | `_sanitize_for_serialization` |
| 171 | 30 | src/oscura/inference/signal_intelligence.py:468 | `check_measurement_suitability` |
| 167 | 28 | src/oscura/analyzers/protocols/i2c.py:126 | `decode` |
| 148 | 28 | src/oscura/analyzers/statistical/checksum.py:187 | `identify_checksum_algorithm` |
| 128 | 28 | src/oscura/loaders/csv_loader.py:166 | `_load_with_pandas` |
| 211 | 27 | src/oscura/analyzers/protocols/swd.py:88 | `decode` |
| 148 | 27 | src/oscura/loaders/touchstone.py:71 | `_parse_touchstone` |
| 196 | 26 | src/oscura/inference/signal_intelligence.py:270 | `assess_signal_quality` |
| 161 | 26 | src/oscura/visualization/protocols.py:605 | `_plot_multi_channel_spi` |
| 145 | 26 | src/oscura/core/extensibility/extensions.py:481 | `select_algorithm` |
| 184 | 25 | src/oscura/analyzers/power/switching.py:50 | `switching_loss` |
| 168 | 25 | src/oscura/analyzers/protocols/uart.py:148 | `decode` |
| 161 | 25 | src/oscura/jupyter/exploratory/legacy.py:481 | `assess_signal_quality` |
| 172 | 24 | src/oscura/jupyter/exploratory/unknown.py:256 | `characterize_unknown_signal` |

## Execution Strategy

### Phase 1: High-Impact Refactoring (Complexity >30)

**Priority**: CRITICAL
**Duration**: ~15 hours
**Functions**: 7 functions

Focus on functions with complexity >30 that pose the highest maintenance risk:

1. `src/oscura/loaders/csv_loader.py:_load_basic` (41 complexity, 150 lines)
2. `src/oscura/workflows/complete_re.py:full_protocol_re` (38 complexity, 302 lines)
3. `src/oscura/guidance/wizard.py:run` (35 complexity, 210 lines)
4. `src/oscura/visualization/power.py:plot_power_profile` (33 complexity, 275 lines)
5. `src/oscura/inference/signal_intelligence.py:classify_signal` (33 complexity, 230 lines)
6. `src/oscura/iot/lorawan/mac_commands.py:parse_mac_command` (33 complexity, 116 lines)
7. `src/oscura/reporting/output.py:_sanitize_for_serialization` (31 complexity, 104 lines)

### Phase 2: Critical Decoders (Complexity 25-30)

**Priority**: HIGH
**Duration**: ~25 hours
**Functions**: ~15 functions

Protocol decoders and signal analysis functions:

- All protocol `decode()` methods (UART, SPI, I2C, SWD)
- Signal intelligence functions
- Checksum identification
- Visualization rendering

### Phase 3: Moderate Complexity (Complexity 20-24)

**Priority**: MEDIUM
**Duration**: ~34 hours
**Functions**: ~70 functions

Batch refactor medium-complexity functions:

- Data loaders
- Analysis workflows
- Export functions
- Configuration handlers

### Phase 4: Long Functions (Complexity <20, Length >100)

**Priority**: LOW
**Duration**: ~50 hours
**Functions**: ~204 functions

Functions that are long but relatively simple:

- Schema definitions
- Registration functions
- Initialization methods
- Plotting setup

## Refactoring Patterns by Function Type

### Pattern 1: Data Loaders

**Example**: `csv_loader.py:_load_basic`

```python
# BEFORE (150 lines, complexity 41)
def _load_basic(self, path):
    # Validation (20 lines)
    # Format detection (30 lines)
    # Data reading (40 lines)
    # Type conversion (30 lines)
    # Error handling (30 lines)

# AFTER (4 functions, each <50 lines, complexity <15)
def _load_basic(self, path):
    self._validate_file(path)
    format_info = self._detect_format(path)
    raw_data = self._read_data(path, format_info)
    return self._convert_types(raw_data, format_info)

def _validate_file(self, path): ...
def _detect_format(self, path): ...
def _read_data(self, path, format_info): ...
def _convert_types(self, raw_data, format_info): ...
```

### Pattern 2: Protocol Decoders

**Example**: `i2c.py:decode`

```python
# BEFORE (167 lines, complexity 28)
def decode(self, data, sample_rate):
    # Validation (15 lines)
    # Clock extraction (40 lines)
    # Data sampling (50 lines)
    # Address/data parsing (40 lines)
    # ACK/NACK detection (22 lines)

# AFTER (5 functions, each <50 lines, complexity <10)
def decode(self, data, sample_rate):
    self._validate_inputs(data, sample_rate)
    clock_edges = self._extract_clock(data)
    samples = self._sample_data_line(data, clock_edges)
    frames = self._parse_frames(samples)
    return self._add_acknowledgments(frames)

def _validate_inputs(self, data, sample_rate): ...
def _extract_clock(self, data): ...
def _sample_data_line(self, data, clock_edges): ...
def _parse_frames(self, samples): ...
def _add_acknowledgments(self, frames): ...
```

### Pattern 3: Visualization

**Example**: `power.py:plot_power_profile`

```python
# BEFORE (275 lines, complexity 33)
def plot_power_profile(data, **kwargs):
    # Parameter validation (30 lines)
    # Figure setup (40 lines)
    # Data preprocessing (50 lines)
    # Multiple subplots (100 lines)
    # Annotations (30 lines)
    # Finalization (25 lines)

# AFTER (7 functions, each <60 lines, complexity <10)
def plot_power_profile(data, **kwargs):
    params = self._validate_and_prepare_params(data, kwargs)
    fig, axes = self._create_figure_layout(params)
    processed = self._preprocess_data(data, params)
    self._plot_power_trace(axes[0], processed, params)
    self._plot_histogram(axes[1], processed, params)
    self._plot_spectrum(axes[2], processed, params)
    return self._finalize_figure(fig, axes, params)
```

### Pattern 4: Schema Registration

**Example**: `config/schema.py:_register_builtin_schemas`

```python
# BEFORE (316 lines, complexity 10)
def _register_builtin_schemas(registry):
    # Protocol schema (100 lines)
    # Pipeline schema (80 lines)
    # Logic family schema (60 lines)
    # Threshold schema (40 lines)
    # Preferences schema (36 lines)

# AFTER (6 functions, each <80 lines, complexity <5)
def _register_builtin_schemas(registry):
    registry.register(_create_protocol_schema())
    registry.register(_create_pipeline_schema())
    registry.register(_create_logic_family_schema())
    registry.register(_create_threshold_profile_schema())
    registry.register(_create_preferences_schema())

def _create_protocol_schema(): ...  # 70 lines
def _create_pipeline_schema(): ...  # 55 lines
def _create_logic_family_schema(): ...  # 50 lines
def _create_threshold_profile_schema(): ...  # 35 lines
def _create_preferences_schema(): ...  # 60 lines
```

## Quality Gates

### After Each Function Refactoring:

1. Run `uv run mypy <file> --strict` → MUST pass
2. Run `uv run ruff check <file>` → MUST show 0 errors
3. Run relevant tests: `./scripts/test.sh -k <module>` → MUST pass
4. Verify line count <100 and complexity <15

### After Each File Refactoring:

1. Run full test suite on file: `./scripts/test.sh tests/unit/<corresponding_test>.py`
2. Run `./scripts/check.sh` → MUST pass
3. Git commit with conventional commit message
4. Update CHANGELOG.md

### After Each Batch (10 functions):

1. Run full test suite: `./scripts/test.sh`
2. Run validators: `python3 .claude/hooks/validate_all.py` → MUST show 5/5
3. Git commit batch with comprehensive message
4. Push to feature branch for CI validation

## Automation Tools

### Analysis Tool

```bash
# Identify all problematic functions
python3 scripts/refactor_long_functions.py --analyze

# Analyze specific file
python3 scripts/refactor_long_functions.py --file src/oscura/config/schema.py --analyze
```

### Batch Refactoring

```bash
# Use code_assistant agent for individual files
/route code_assistant "Refactor src/oscura/loaders/csv_loader.py:_load_basic - \
  extract validation, format detection, data reading, and type conversion into \
  separate methods"

# Test after refactoring
./scripts/test.sh -k csv_loader

# Commit if passing
git add src/oscura/loaders/csv_loader.py tests/
git commit -m "refactor(loaders): decompose csv_loader._load_basic (complexity 41→<15)"
```

## Progress Tracking

Create batches of 10 functions, refactor systematically:

### Batch 1 (Complexity >30) - PRIORITY

- [ ] `csv_loader.py:_load_basic` (41)
- [ ] `complete_re.py:full_protocol_re` (38)
- [ ] `wizard.py:run` (35)
- [ ] `power.py:plot_power_profile` (33)
- [ ] `signal_intelligence.py:classify_signal` (33)
- [ ] `mac_commands.py:parse_mac_command` (33)
- [ ] `output.py:_sanitize_for_serialization` (31)
- [ ] `signal_intelligence.py:check_measurement_suitability` (30)
- [ ] `i2c.py:decode` (28)
- [ ] `checksum.py:identify_checksum_algorithm` (28)

### Batch 2 (Complexity 25-28)

- [ ] `csv_loader.py:_load_with_pandas` (28)
- [ ] `swd.py:decode` (27)
- [ ] `touchstone.py:_parse_touchstone` (27)
- [ ] `signal_intelligence.py:assess_signal_quality` (26)
- [ ] `protocols.py:_plot_multi_channel_spi` (26)
- [ ] `extensions.py:select_algorithm` (26)
- [ ] `switching.py:switching_loss` (25)
- [ ] `uart.py:decode` (25)
- [ ] `legacy.py:assess_signal_quality` (25)
- [ ] Continue through all complexity 25-28 functions...

### Batches 3-30 (Medium/Low Priority)

Continue with remaining 276 functions in batches of 10.

## Success Metrics

- [ ] All 296 functions <100 lines
- [ ] All 296 functions complexity <15
- [ ] All tests passing (`./scripts/test.sh`)
- [ ] All validators passing (`validate_all.py`)
- [ ] No regressions in code coverage
- [ ] CHANGELOG.md updated for all changes
- [ ] Feature branch merged to main after full CI passes

## Timeline

- **Week 1-2**: Phase 1 (Critical complexity >30) - 7 functions
- **Week 3-5**: Phase 2 (Decoders, complexity 25-30) - 15 functions
- **Week 6-10**: Phase 3 (Medium complexity 20-24) - 70 functions
- **Week 11-20**: Phase 4 (Long but simple) - 204 functions

**Total**: 20 weeks (5 months) at 15 minutes per function average

## Risk Mitigation

1. **Breaking Changes**: All refactoring maintains identical public APIs
2. **Test Coverage**: Run tests after every file, batch, and phase
3. **Rollback Strategy**: Git commits per file allow easy rollback
4. **CI Integration**: Feature branch tested in CI before merging
5. **Code Review**: Each batch reviewed before moving to next

## Next Steps

1. Review and approve this plan
2. Create feature branch: `refactor/long-functions-batch-1`
3. Start with Batch 1 (highest complexity functions)
4. Use code_assistant agent for each function
5. Commit after each successful refactoring
6. Track progress in this document
