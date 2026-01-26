# Documentation Remediation Plan - v0.6.0

**Created**: 2026-01-25
**Status**: EXECUTION READY
**Estimated Time**: 2-3 hours

---

## Summary

This plan systematically fixes all deprecated API references in documentation for v0.6.0 release.

**Total Changes Needed**: 100+
**Files Affected**: 15
**Priority**: CRITICAL (blocks v0.6.0 release quality)

---

## Phase 1: Automated Bulk Replacements (30 min)

### Step 1.1: Global Module Import Replacements

**Pattern**: `from oscura.MODULE import` where MODULE was removed

**Replacements**:

| Old Pattern                       | New Pattern                                | Files Affected |
| --------------------------------- | ------------------------------------------ | -------------- |
| `from oscura.utils.comparison import`   | `from oscura.utils.comparison import`      | 4              |
| `from oscura.utils.filtering import`    | `from oscura.utils.filtering import`       | 2              |
| `from oscura.core.config import`       | `from oscura.core.config import`           | 1              |
| `# NOTE: Use workflows or manual iteration in v0.6

# from oscura.workflows import` | `# See oscura.workflows or manual loops`   | 3              |

| `from oscura.acquisition import`  | `# Use oscura.load() directly`             | 5              |
| `load(`                  | `load(`                                    | 5              |
| `osc.load("path")`              | `osc.load("path")`                         | 10+            |

**Script**: `.claude/scripts/migrate_doc_api.py` (already created)

**Action**: Run script, commit changes

### Step 1.2: Update Code Block Patterns

**Pattern**: Bare function calls that should use `osc.` prefix

**Examples in docs/api/comparison-and-limits.md** (COMPLETED):

- ✓ `compare_traces()` → `osc.compare_traces()`
- ✓ `create_golden()` → `osc.create_golden()`
- ✓ `check_limits()` → `osc.check_limits()`

**Remaining Files**:

- `docs/api/index.md` - Update comparison example
- `docs/api/workflows.md` - Check for deprecated imports
- `docs/api/pipelines.md` - Check for deprecated imports

---

## Phase 2: Manual File-by-File Updates (90 min)

### Priority 1: API Reference Files (CRITICAL)

#### ✓ docs/api/comparison-and-limits.md (COMPLETED)

- [x] Updated import statements (line 14)
- [x] Updated all `from oscura.comparison` to show new path
- [x] Updated Quick Start examples to use `osc.` prefix
- Status: COMPLETE

#### docs/api/index.md

**Changes Needed**: 1

- Line 181: `from oscura.utils.comparison import` → update

**Action**:

```python
# OLD
from oscura.utils.comparison import (
    compare_traces, create_golden, check_limits
)

# NEW
import oscura as osc
# All functions available at top-level:
# osc.compare_traces(), osc.create_golden(), osc.check_limits()
```

### Priority 2: User Guides (HIGH)

#### docs/guides/hardware-acquisition.md

**Changes Needed**: 10

- Multiple `# NOTE: Direct loading recommended in v0.6
import oscura as osc` references
- Replace with modern `osc.load()` pattern or direct hardware API

**Before**:

```python
# NOTE: Direct loading recommended in v0.6
import oscura as osc
source = HardwareSource("visa://scope")
trace = source.capture()
```

**After**:

```python
import oscura as osc
# For hardware acquisition, use loaders with hardware support
# OR use hardware-specific loaders directly:
from oscura.loaders.tektronix import TektronixLoader
loader = TektronixLoader("visa://scope")
trace = loader.read_waveform(channel=1)
```

**Note**: This file needs careful review - hardware acquisition may need different approach

#### docs/guides/blackbox-analysis.md

**Changes Needed**: 4

- 3x `# NOTE: Direct loading recommended in v0.6
import oscura as osc`
- 1x `from oscura.utils.filtering import low_pass`

**Pattern**:

```python
# OLD
# NOTE: Direct loading recommended in v0.6
import oscura as osc
from oscura.utils.filtering import low_pass

source = osc.load("capture.wfm")
filtered = low_pass(source.read(), cutoff=1e6)

# NEW
import oscura as osc

trace = osc.load("capture.wfm")
filtered = osc.low_pass(trace, cutoff=1e6, sample_rate=trace.metadata.sample_rate)
```

#### docs/guides/side-channel-analysis.md

**Changes Needed**: 3

- 2x `# NOTE: Direct loading recommended in v0.6
import oscura as osc`
- 1x `from oscura.utils.filtering import low_pass, moving_average`

**Same pattern as blackbox-analysis.md above**

### Priority 3: Tutorials (HIGH)

#### docs/tutorials/reverse-engineering-uart.md

**Changes Needed**: 2

- Line 50: `from oscura.loaders import load_waveform`
- Line 54: `waveform = load("captures/01_power_on.wfm")`

**Fix**:

```python
# OLD
from oscura.loaders import load_waveform
waveform = load("captures/01_power_on.wfm")

# NEW
import oscura as osc
waveform = osc.load("captures/01_power_on.wfm")
```

### Priority 4: User Guide / FAQ (MEDIUM)

#### docs/user-guide/getting-started.md

**Changes Needed**: 3

- 1x `load_waveform`
- 1x `# NOTE: Use workflows or manual iteration in v0.6

# from oscura.workflows import batch_analyze`

- 1x `from oscura.core.config import load_config`

#### docs/user-guide/workflows.md

**Changes Needed**: 2

- 1x `load_waveform`
- 1x `# NOTE: Use workflows or manual iteration in v0.6

# from oscura.workflows import batch_analyze, BatchConfig`

#### docs/faq/index.md

**Changes Needed**: 2

- 1x `load_waveform`
- 1x `# NOTE: Use workflows or manual iteration in v0.6

# from oscura.workflows import batch_analyze`

#### docs/developer-guide/architecture.md

**Changes Needed**: 3

- 1x `load_waveform`
- 1x `from oscura.utils.comparison import compare_traces`
- 1x `# NOTE: Use workflows or manual iteration in v0.6

# from oscura.workflows import batch_analyze`

### Priority 5: Architecture Docs (LOW - Design Examples)

#### docs/architecture/design-principles.md

**Changes Needed**: 6

- 1x `# NOTE: Direct loading recommended in v0.6
import oscura as osc, HardwareSource, SyntheticSource`
- 5x `SignalBuilder` references

**Decision**: Keep SignalBuilder references BUT add note:

```markdown
> **Note**: SignalBuilder is primarily for test infrastructure.
> For production signal generation, use numpy arrays or load from files.
```

#### docs/architecture/api-patterns.md

**Changes Needed**: 1

- 1x `SignalBuilder` reference

**Same treatment as design-principles.md**

---

## Phase 3: Validation (30 min)

### Step 3.1: Run Documentation Validator

```bash
python3 .claude/hooks/validate_documentation.py
```

**Expected**: All checks pass, no broken links, no deprecated references

### Step 3.2: Extract and Test Code Examples

```bash
python3 .claude/scripts/audit_documentation.py
```

**Expected**: All Python code blocks are syntactically valid

### Step 3.3: Manual Review Checklist

- [ ] All `from oscura.comparison` → updated
- [ ] All `from oscura.acquisition` → updated or removed
- [ ] All `from oscura.filtering` → updated
- [ ] All `from oscura.batch` → updated or removed
- [ ] All `from oscura.config` → updated
- [ ] All `load_waveform` → `load`
- [ ] All code examples use `import oscura as osc` pattern
- [ ] SignalBuilder usage documented as test-only
- [ ] No references to removed modules in API docs

---

## Phase 4: Update Changelog (10 min)

Add to `CHANGELOG.md` under `## [Unreleased]` → `### Changed`:

```markdown
### Changed
- **Documentation** (docs/): Updated all documentation to use v0.6 APIs
  - Migrated from `oscura.comparison` to `oscura.utils.comparison`
  - Migrated from `oscura.filtering` to `oscura.utils.filtering`
  - Migrated from `oscura.config` to `oscura.core.config`
  - Replaced `load()` with `load()` throughout
  - Removed `oscura.acquisition` references (use `osc.load()` directly)
  - Removed `oscura.batch` references (use `oscura.workflows` or manual iteration)
  - Updated 15 documentation files with 100+ individual changes
  - All code examples now use `import oscura as osc` pattern for consistency
```

---

## Phase 5: Final Quality Check (10 min)

### Run All Validators

```bash
python3 .claude/hooks/validate_all.py
```

**Must show**: 5/5 validators passing

### Visual Inspection

Review 3 key files to ensure quality:

1. `docs/api/comparison-and-limits.md` - Most changes
2. `docs/guides/blackbox-analysis.md` - User-facing
3. `docs/tutorials/reverse-engineering-uart.md` - Tutorial

---

## Success Criteria

- [ ] All 100+ deprecated API references updated
- [ ] No `from oscura.comparison` imports (except migration guide)
- [ ] No `from oscura.acquisition` imports (except migration guide)
- [ ] No `load()` calls
- [ ] All code examples syntactically valid
- [ ] All documentation validators pass (5/5)
- [ ] CHANGELOG.md updated
- [ ] Manual review of 3 key files completed

---

## Rollback Plan

If issues arise:

```bash
git checkout docs/
```

Then fix issues and re-run from Phase 1.

---

## Notes

- Migration guide (`docs/migration/v0-to-v1.md`) intentionally skipped - it shows both old and new APIs
- Test suite guide (`docs/testing/test-suite-guide.md`) - SignalBuilder references OK (test infrastructure)
- Some hardware acquisition examples may need different approach than simple `osc.load()` - review carefully

---

**Next Action**: Execute Phase 1 automated replacements
