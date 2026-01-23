# Oscura Demonstrations: Comprehensive Improvement Roadmap

**Date:** 2026-01-23
**Analysis:** Deep audit of all 112 demonstrations covering code quality, documentation SSOT, and usability
**Status:** Implementation plan with prioritized actions

---

## EXECUTIVE SUMMARY

Comprehensive analysis of 112 demonstrations identified:
- **8-10 critical code issues** (hardcoded values, missing validation)
- **7 SSOT violations** (~1200 duplicate lines in documentation)
- **~4500 lines of excessive verbosity** (75% reduction possible)
- **Missing feature:** Command-line data file specification support

**Overall Assessment:** Demonstrations are **production-quality** but have systematic patterns that should be addressed for optimal robustness and maintainability.

---

## PART 1: CRITICAL CODE FIXES (HIGH PRIORITY)

### 1.1 Add Command-Line Data File Support to BaseDemo

**Status:** ✓ **COMPLETED** (2026-01-23)

**Problem:** Users cannot specify custom data files to experiment with demonstrations

**Solution:** Enhanced BaseDemo with `--data-file` argument support

**Implementation Details:**
- Added argparse to BaseDemo.execute() method
- Created load_custom_data() method supporting NPZ format
- Maintained backward compatibility (all 112 demos validated 100% pass rate)
- Updated CHANGELOG.md with user-facing feature description

**Impact:** ALL 112 demonstrations now support custom data experimentation

**Usage:**
```bash
# Use demonstration's built-in test data
python demonstrations/02_basic_analysis/01_waveform_measurements.py

# Use custom data file
python demonstrations/02_basic_analysis/01_waveform_measurements.py --data-file my_capture.npz

# Show help including new argument
python demonstrations/02_basic_analysis/01_waveform_measurements.py --help
```

---

### 1.2 Fix Hardcoded Validation Values

**Status:** ⏳ PLANNED

**Problem:** Demonstrations use magic numbers in validation that break when parameters change

**Files Affected:**
- `02_basic_analysis/01_waveform_measurements.py` (lines 296-315)
- `02_basic_analysis/02_statistics.py` (lines 294-320)
- `01_data_loading/02_logic_analyzers.py` (lines 376-381)

**Example Issue:**
```python
# CURRENT - BRITTLE
if not validate_approximately(results["pulse_rise_time"], 784e-9, tolerance=0.1):
    # 784e-9 is hardcoded, breaks if sample_rate changes
```

**Recommended Fix:**
```python
# IMPROVED - ROBUST
expected_rise_time = calculate_sampling_limited_value(
    nominal=10e-9,
    sample_rate=data["pulse_train"].metadata.sample_rate
)
if not validate_approximately(results["pulse_rise_time"], expected_rise_time, tolerance=0.1):
```

**Priority:** HIGH
**Estimated Effort:** 4-6 hours
**Files to Fix:** 8-10 demonstrations

---

### 1.3 Add Numpy Random Seeds for Deterministic Tests

**Status:** ⏳ PLANNED

**Problem:** Demonstrations using `np.random` without seeds cause non-deterministic behavior and flaky tests

**Files Affected:**
- `02_basic_analysis/02_statistics.py` (line 100)
- `04_advanced_analysis/01_jitter_analysis.py` (line 156)
- `01_data_loading/03_automotive_formats.py` (various)
- 15+ other demonstrations

**Recommended Fix:**
```python
def generate_test_data(self) -> dict[str, Any]:
    # Add at start of ALL demonstrations using np.random
    np.random.seed(42)  # Deterministic randomness for reproducible tests

    # ... rest of data generation
```

**Priority:** HIGH (causes test flakiness)
**Estimated Effort:** 2-3 hours (bulk find/replace with verification)
**Files to Fix:** ~20 demonstrations

---

### 1.4 Fix Custom Binary Offset Calculation

**Status:** ⏳ PLANNED

**Problem:** Incorrect offset calculation assumes 8-byte data type

**File:** `demonstrations/01_data_loading/05_custom_binary.py` (line 318)

**Current:**
```python
offset=file_info["header_size"] // 8,  # Assumes 8-byte floats
```

**Fix:**
```python
dtype_size = np.dtype(f"{endian_char}f8").itemsize
offset=file_info["header_size"] // dtype_size,
```

**Priority:** MEDIUM
**Estimated Effort:** 15 minutes
**Files to Fix:** 1 demonstration

---

### 1.5 Create Common Constants File

**Status:** ⏳ PLANNED

**Problem:** Magic numbers repeated across 40+ demonstrations

**Common Values:**
- `0.05` (5% tolerance) - used 15+ times
- `0.01` (1% tolerance) - used 10+ times
- `1e-14` (float epsilon) - used 5+ times
- `0.707` (1/√2 for RMS) - used 3 times

**Solution:** Create `demonstrations/common/constants.py`

```python
"""Common constants used across demonstrations."""

import numpy as np

# Validation tolerances
TOLERANCE_STRICT = 0.01    # 1% - for precise measurements
TOLERANCE_NORMAL = 0.05    # 5% - for typical measurements
TOLERANCE_RELAXED = 0.10   # 10% - for noisy/derived measurements

# Numerical precision
FLOAT_EPSILON = 1e-14      # Float comparison threshold
FLOAT_TOLERANCE = 1e-6     # Relative tolerance for float comparisons

# Mathematical constants
SINE_RMS_FACTOR = 1 / np.sqrt(2)  # 0.707... (sine peak to RMS)
SQRT2 = np.sqrt(2)                # 1.414...
```

**Priority:** MEDIUM
**Estimated Effort:** 3-4 hours (create file + update ~40 demonstrations)
**Files to Fix:** 40+ demonstrations

---

### 1.6 Consolidate Edge Detection Logic

**Status:** ⏳ PLANNED

**Problem:** Edge detection code duplicated across 5+ demonstrations

**Files Affected:**
- `04_advanced_analysis/01_jitter_analysis.py` (30+ lines)
- `02_basic_analysis/01_waveform_measurements.py`
- `01_data_loading/02_logic_analyzers.py`
- Others

**Solution:** Create `demonstrations/common/signal_processing.py`

```python
"""Common signal processing utilities for demonstrations."""

def find_edges(
    trace: WaveformTrace,
    edge_type: str = 'rising',
    threshold: float | None = None,
    interpolate: bool = True
) -> np.ndarray:
    """Universal edge detection with sub-sample interpolation.

    Args:
        trace: Input waveform
        edge_type: 'rising', 'falling', or 'both'
        threshold: Detection threshold (auto if None)
        interpolate: Use sub-sample interpolation for precision

    Returns:
        Array of edge timestamps
    """
    # Single implementation used by all demos
```

**Priority:** LOW (code duplication, not a bug)
**Estimated Effort:** 4-5 hours
**Files to Fix:** 5+ demonstrations

---

## PART 2: DOCUMENTATION SSOT FIXES (HIGH PRIORITY)

### 2.1 Remove Duplicate Installation Instructions

**Status:** ⏳ PLANNED

**Problem:** Installation instructions duplicated in ALL 19 category READMEs (~30 lines × 19 = 570 duplicate lines)

**Files Affected:** All `demonstrations/XX_*/README.md` files (lines 11-40 typical)

**Solution:**
```markdown
<!-- BEFORE (repeated 19 times): -->
## Prerequisites

Python 3.12+ with:
```bash
pip install oscura
# ... 25 more lines ...
```

<!-- AFTER (single reference): -->
## Prerequisites

See [main README.md](../README.md#installation) for installation instructions.
```

**Priority:** HIGH (massive duplication)
**Estimated Effort:** 1 hour (bulk edit 19 files)
**Lines Removed:** ~570

---

### 2.2 Remove Duplicate "Running Demonstrations" Instructions

**Status:** ⏳ PLANNED

**Problem:** Identical 35-line "How to Run" section in ALL 19 category READMEs (35 × 19 = 665 duplicate lines)

**Files Affected:** All `demonstrations/XX_*/README.md` files (lines 129-164 typical)

**Solution:**
```markdown
<!-- BEFORE (repeated 19 times): -->
## Running the Demonstrations

### Option 1: Direct Execution
...35 lines of instructions...

<!-- AFTER (single reference): -->
## Running the Demonstrations

See [main README.md](../README.md#running-demonstrations) for execution options.

Category-specific note: Start with `00_hello_world.py` before trying advanced demos.
```

**Priority:** HIGH (massive duplication)
**Estimated Effort:** 1 hour
**Lines Removed:** ~630

---

### 2.3 Consolidate API Coverage Statistics

**Status:** ⏳ PLANNED

**Problem:** Three different numbers for "total API capabilities"
- `README.md`: "813 API symbols"
- `STATUS.md`: "266 API symbols"
- `CAPABILITY_CROSSREF.md`: "201 capabilities"

**Solution:**
1. Clarify definitions in STATUS.md:
   - 266 = user-facing symbols in `__all__` (authoritative)
   - 813 = includes internal symbols (not relevant to users)
   - 201 = capabilities tracked in demonstrations (subset of 266)

2. Update README.md to use 266 consistently

3. Generate statistics from `capability_index.py` instead of hardcoding

**Priority:** HIGH (user confusion)
**Estimated Effort:** 30 minutes
**Files to Fix:** 3 documentation files

---

### 2.4 Remove Excessive Verbosity from Category READMEs

**Status:** ⏳ PLANNED

**Problem:** ~4500 lines of verbose sections that add little value:
- "What You'll Learn" (50-80 lines per README, duplicates demo docstrings)
- "Tips for Learning" (60 lines of generic advice)
- "Common Issues" (40 lines of troubleshooting per category)
- "Resources" (30 lines of obvious information)

**Solution:** Reduce each category README from ~500 lines to ~150 lines (70% reduction)

**Example Reduction:**
```markdown
<!-- BEFORE: 75 lines -->
## What You'll Learn

### 1. Waveform Loading and Processing
Learn how to load waveform data from multiple sources...
[68 more lines of content already in demo docstrings]

<!-- AFTER: 10 lines -->
## What You'll Learn

This category covers waveform loading (VCD, WAV, CSV, binary formats) and basic measurements
(amplitude, frequency, rise time, RMS). Each demonstration includes complete examples with
validation. See individual demo files for detailed learning outcomes.
```

**Priority:** MEDIUM (improves scannability but not functionally broken)
**Estimated Effort:** 6-8 hours (19 READMEs × 20-30 min each)
**Lines Removed:** ~3500

---

### 2.5 Create Centralized TROUBLESHOOTING.md

**Status:** ⏳ PLANNED

**Problem:** Troubleshooting scattered across 15+ category READMEs (40 lines each = 600 duplicate lines)

**Solution:** Create `demonstrations/TROUBLESHOOTING.md` consolidating all common issues:

```markdown
# Oscura Demonstrations: Troubleshooting Guide

## File Format Issues

### VCD Files Not Loading
**Symptom:** "Invalid VCD header" error
**Causes:** ...
**Solutions:** ...

[Continue for all common issues consolidated from category READMEs]
```

**Priority:** MEDIUM
**Estimated Effort:** 2 hours (consolidate + update 15 READMEs)
**Lines Removed:** ~550

---

## PART 3: MISSING DOCUMENTATION (LOW PRIORITY)

### 3.1 Create ARCHITECTURE.md

**Status:** ⏳ PLANNED

**Purpose:** Explain BaseDemo pattern, validation system, capability tracking

**Content:**
- BaseDemo template pattern and lifecycle
- Validation framework (validate_approximately, validate_range, etc.)
- Capability indexing system
- How to add new demonstrations
- Best practices and patterns

**Priority:** LOW (mainly for contributors)
**Estimated Effort:** 2-3 hours
**Lines to Add:** ~300

---

### 3.2 Create WORKFLOWS.md

**Status:** ⏳ PLANNED

**Purpose:** Complete use-case workflows for common reverse engineering tasks

**Content:**
- "Reverse engineer unknown serial protocol" → demos 01, 03, 06, 16
- "Analyze automotive diagnostic bus" → demos 01, 03, 05, 16
- "Validate signal integrity compliance" → demos 02, 04, 19
- "Debug power supply ripple" → demos 01, 02, 04
- [5-10 complete workflows total]

**Priority:** LOW (nice to have)
**Estimated Effort:** 2-3 hours
**Lines to Add:** ~400

---

## PART 4: CODE ENHANCEMENTS (LOW PRIORITY)

### 4.1 Standardize Error Messages

**Status:** ⏳ PLANNED

**Problem:** Inconsistent error message formats across demonstrations

**Solution:** Add to BaseDemo or common/validation.py:

```python
def validation_error_message(
    metric: str,
    expected: float,
    actual: float,
    tolerance: float,
    units: str = ""
) -> str:
    """Generate standardized validation error message."""
    expected_min = expected * (1 - tolerance)
    expected_max = expected * (1 + tolerance)
    return (
        f"{metric}: {actual:.6g}{units} outside expected range "
        f"[{expected_min:.6g}, {expected_max:.6g}]{units} "
        f"(expected {expected:.6g} ± {tolerance * 100:.1f}%)"
    )
```

**Priority:** LOW
**Estimated Effort:** 1-2 hours
**Files to Fix:** 30+ demonstrations

---

### 4.2 Add Missing Edge Case Handling

**Status:** ⏳ PLANNED

**Files:** `02_basic_analysis/02_statistics.py` (peak detection)

**Issue:** Doesn't handle cases where <2 peaks found, >2 peaks found, or peaks too close

**Solution:** Add scipy.signal.find_peaks with proper validation

**Priority:** LOW (only affects specific demo)
**Estimated Effort:** 1 hour
**Files to Fix:** 1 demonstration

---

## IMPLEMENTATION PRIORITY SUMMARY

### CRITICAL (Do First - Total: ~8 hours)
1. ✓ Add command-line data file support to BaseDemo (DONE)
2. Add numpy random seeds (~2-3 hours)
3. Fix hardcoded validation values (~4-6 hours)

### HIGH PRIORITY (Next - Total: ~4 hours)
4. Remove duplicate installation instructions (~1 hour)
5. Remove duplicate running instructions (~1 hour)
6. Consolidate API coverage statistics (~30 min)
7. Fix custom binary offset bug (~15 min)
8. Create common constants file (~3 hours)

### MEDIUM PRIORITY (After High - Total: ~12 hours)
9. Reduce category README verbosity (~6-8 hours)
10. Create centralized TROUBLESHOOTING.md (~2 hours)
11. Consolidate edge detection logic (~4 hours)

### LOW PRIORITY (Optional - Total: ~10 hours)
12. Create ARCHITECTURE.md (~2-3 hours)
13. Create WORKFLOWS.md (~2-3 hours)
14. Standardize error messages (~1-2 hours)
15. Add missing edge case handling (~1 hour)
16. Various minor improvements (~3 hours)

**Total Estimated Effort:** ~34 hours for complete implementation

---

## TESTING STRATEGY

After each phase:
1. Run `python3 demonstrations/validate_all.py` (must maintain 100% pass rate)
2. Run `./scripts/check.sh` (must pass linting/type checking)
3. Run `python3 .claude/hooks/validate_all.py` (must pass all validators)
4. Spot-check 5-10 demonstrations manually for correctness

---

## CHANGE TRACKING

### Completed Changes
- ✓ 2026-01-23: Added command-line data file support to BaseDemo - ALL 112 demonstrations now accept `--data-file` argument for custom NPZ data experimentation (commit pending)
- ✓ 2026-01-23: Tightened jitter analysis validation thresholds (commit 492a0ba)
- ✓ 2026-01-23: Enhanced waveform measurement documentation (commit 492a0ba)
- ✓ 2026-01-23: Fixed spurious .claude directory creation (commit e8ba94e)
- ✓ 2026-01-23: Cleaned up vestigial demo directories (commits 60f0b7a, bd89f53)

### In Progress
- None (ready to commit command-line data file support)

### Next Up
- ⏳ Adding numpy random seeds to ~20 demonstrations
- ⏳ Creating common constants file for magic numbers
- ⏳ Fixing hardcoded validation values in 8-10 demonstrations

### Planned
- ⏳ All items listed above

---

## NOTES FOR FUTURE MAINTAINERS

### Key Principles
1. **SSOT:** Avoid duplicating information - reference authoritative source
2. **DRY Code:** Extract common patterns (edge detection, validation, etc.)
3. **Determinism:** Always seed random number generators in demonstrations
4. **Robustness:** Derive expected values from generation parameters, not hardcoded
5. **Clarity:** Keep documentation focused - remove verbosity that doesn't add clarity

### Common Patterns to Watch For
- Hardcoded validation values (should be computed from data generation params)
- Missing `np.random.seed()` calls (causes flaky tests)
- Duplicate documentation across category READMEs (should reference main README)
- Magic numbers (should be named constants)
- Verbose "What You'll Learn" sections (should be 5-10 lines, not 50-80)

### Before Adding New Demonstrations
1. Use BaseDemo template
2. Support --data-file argument (automatic via BaseDemo)
3. Seed random number generators if using randomness
4. Use common constants instead of magic numbers
5. Derive validation values from generation parameters
6. Keep docstring focused (10-20 lines, not 100+)
7. Add to capability_index.py tracking

---

**END OF ROADMAP**

*This document represents comprehensive analysis of 112 demonstrations. It prioritizes improvements by impact and provides specific recommendations with code examples for implementation.*
