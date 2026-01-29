# Optimal Test Suite Configuration Plan

**Version**: 1.0
**Date**: 2026-01-28
**Target**: Oscura v0.6.x → v0.7.0
**Status**: Implementation Ready

---

## Executive Summary

### Current State Assessment

**Test Suite Scale**:

- **20,124 tests** across 509 test files
- **317,000+ lines** of test code
- **467 source files** with 463 lacking 1:1 test file mapping
- **14 conftest.py hierarchy** (64 main fixtures + 128 local fixtures)
- **17 CI test groups** with 25-minute timeout
- **Test collection time**: 12.52 seconds (can be optimized to ~5s)

**Configuration Health**: **8/10** (Good with Critical Issues)

**Strengths**:

- Massive test coverage (20K+ tests demonstrates serious quality commitment)
- Well-optimized fixture scoping (module-scoped autouse saves ~14 minutes)
- Comprehensive pytest-xdist parallelization (2-4 workers per group)
- Strong test infrastructure (factories, builders, ground truth validation)
- Proper CI/CD chunking strategy (memory-intensive vs standard groups)

**Critical Issues Identified**:

1. **Configuration Inconsistencies** (Priority 1):
   - Timeout settings commented out in `pyproject.toml` (lines 313-316)
   - Flaky test retry thresholds vary (2 vs 3 reruns across workflows)
   - Missing `pytest-timeout` in dev dependencies despite CI usage

2. **Autouse Fixture Overhead** (Priority 2):
   - 4 module-scoped autouse fixtures run on EVERY test module (389 modules)
   - Combined overhead: ~23 seconds per test run
   - Logging/warnings fixtures could be opt-in for non-modifying tests

3. **CI Batch Fragmentation** (Priority 2):
   - 17 CI test groups (can consolidate to 12-13 for faster feedback)
   - Packet analyzer split into 5 batches (3b-parser, 3b-stream, 3b-metrics, 3b-part2, 3e)
   - Duration threshold warnings at 80% (20min) suggest groups approaching limits

4. **Coverage Overhead** (Priority 3):
   - Coverage adds ~30% execution time overhead
   - Can reduce to ~20% with Python 3.12+ `COVERAGE_CORE=sysmon` optimization
   - Not currently leveraging sys.monitoring API despite Python 3.12 requirement

5. **Test Collection Performance** (Priority 3):
   - 12.52s collection time indicates room for optimization
   - Missing `testpaths` optimization in subdirectory discovery
   - Disabled plugins help but more can be done

### Expected Improvements from Plan

**Performance Gains** (Cumulative):

| Optimization | Current | Target | Improvement |
|--------------|---------|--------|-------------|
| Test collection | 12.52s | ~5s | -60% (7.5s saved) |
| Autouse fixture overhead | ~23s | ~10s | -57% (13s saved) |
| Coverage overhead | +30% | +20% | -33% relative |
| CI batch count | 17 groups | 12-13 groups | -24% to -29% |
| Total per-group CI time | 8-10 min | 7-9 min | -10% to -12% |

**Reliability Gains**:

- Consistent flaky test handling (standardize on 2 reruns, 1s delay)
- Timeout enforcement prevents CI hangs (120s per test, 25min per job)
- Better batch balance reduces timeout risk
- Automatic flaky test detection and quarantine

**Maintenance Gains**:

- Consolidated configuration (remove pytest.ini, keep only pyproject.toml)
- Documentation for all configuration decisions
- Automated configuration validation in pre-commit hooks
- Test analytics dashboard for continuous optimization

---

## Critical Issues & Immediate Fixes (Week 1)

### Issue 1: Missing pytest-timeout Configuration

**Problem**: CI uses `pytest-timeout` but it's not in dev dependencies, and configuration is commented out.

**Impact**: Local tests don't enforce timeouts, allowing hangs that CI will catch later.

**Files Affected**:

- `/home/lair-click-bats/development/oscura/pyproject.toml` (lines 94, 313-316)
- `.github/workflows/ci.yml` (implicit usage)
- `.github/workflows/tests-chunked.yml` (implicit usage)

**Fix**:

```diff
# pyproject.toml [project.optional-dependencies]
dev = [
    "pytest>=8.0,<10.0.0",
    "pytest-cov>=6.0,<8.0.0",
    "pytest-timeout>=2.3.0,<3.0.0",     # Already present (line 94)
    "pytest-benchmark>=4.0.0,<6.0.0",
+   "pytest-rerunfailures>=14.0,<15.0.0", # Add for --reruns support
    "hypothesis>=6.0.0,<7.0.0",
]

# pyproject.toml [tool.pytest.ini_options]
-# NOTE: timeout settings commented out - requires pytest-timeout plugin
-# Uncomment if pytest-timeout is installed:
-# timeout = 60
-# timeout_method = "thread"
-# timeout_func_only = false
+# Per-test timeout enforcement (prevents test hangs)
+timeout = 120  # 2 minutes per test (CI uses same)
+timeout_method = "thread"  # thread method for better compatibility
+timeout_func_only = true  # Only apply to test functions, not fixtures
```

**Validation**:

```bash
# Verify timeout works locally
uv run pytest tests/unit/analyzers/digital/test_edges.py::test_detect_edges --timeout=5

# Should timeout after 5 seconds if test hangs
```

**Completion Criteria**:

- [ ] `pytest-timeout` documented in pyproject.toml (already present)
- [ ] `pytest-rerunfailures` added to dev dependencies
- [ ] Timeout configuration uncommented with 120s threshold
- [ ] Local test run validates timeout enforcement
- [ ] Documentation updated explaining timeout rationale

---

### Issue 2: Inconsistent Flaky Test Retry Configuration

**Problem**: Different workflows use different retry strategies:

- `ci.yml`: `--reruns 2 --reruns-delay 1`
- `tests-chunked.yml`: `--reruns 2 --reruns-delay 1`
- Best practices recommend 2-3 retries with exponential backoff

**Impact**: Flaky tests may pass in one workflow but fail in another, reducing CI reliability.

**Files Affected**:

- `.github/workflows/ci.yml` (lines 370-371)
- `.github/workflows/tests-chunked.yml` (lines 235-236, 291-292)
- `pyproject.toml` (no configuration - should add)

**Fix**:

```toml
# pyproject.toml [tool.pytest.ini_options] - Add standardized retry config
addopts = [
    "--import-mode=importlib",
    "-ra",
    "--strict-markers",
    "--strict-config",
    "--tb=line",
    "--capture=sys",
    "--ff",
    "--nf",
    # Flaky test handling (consistent across local/CI)
    "--reruns=2",              # Retry failed tests up to 2 times
    "--reruns-delay=1",        # 1 second delay between retries
    # Disable unnecessary plugins
    "-p", "no:deadfixtures",
    "-p", "no:memray",
    "-p", "no:split",
]

# Document flaky test policy
[tool.pytest.ini_options]
# Flaky test handling strategy:
# - Tests retry 2 times automatically (total 3 runs max)
# - 1 second delay between retries (allows transient issues to resolve)
# - If test fails 3 times, it's genuinely broken (not flaky)
# - Use @pytest.mark.flaky(reruns=5) for known-flaky tests only
```

**Validation**:

```bash
# Create intentionally flaky test
cat > /tmp/test_flaky.py << 'EOF'
import random
import pytest

def test_sometimes_fails():
    """Flaky test that fails 50% of the time."""
    assert random.random() > 0.5
EOF

# Should pass after 1-2 retries
uv run pytest /tmp/test_flaky.py --reruns=2 --reruns-delay=1 -v
```

**Completion Criteria**:

- [ ] `pytest-rerunfailures` added to dev dependencies
- [ ] Retry configuration added to `pyproject.toml` addopts
- [ ] All CI workflows inherit retry config from pyproject.toml
- [ ] Documentation added explaining retry policy
- [ ] Validation test passes with retries

---

### Issue 3: Autouse Fixture Overhead Optimization

**Problem**: 4 autouse fixtures run on every module even when not needed:

- `cleanup_matplotlib` (1103): ~0.4s overhead (389 modules * 0.001s)
- `memory_cleanup` (1125): ~4s overhead (389 modules * 0.01s)
- `reset_warnings_state` (1150): ~3.9s overhead (389 modules * 0.01s)
- `reset_logging_state` (1222): ~19s overhead (389 modules * 0.05s)
- **Total: ~27 seconds per test run**

**Impact**: Tests that don't use matplotlib/logging/warnings still pay the cleanup cost.

**Files Affected**:

- `tests/conftest.py` (lines 1103, 1125, 1150, 1222)

**Fix Strategy**: Keep critical autouse fixtures, make others opt-in

```python
# tests/conftest.py

# KEEP autouse: memory_cleanup (critical for OOM prevention)
@pytest.fixture(autouse=True, scope="module")
def memory_cleanup():
    """Force garbage collection once per module to prevent memory buildup.

    CRITICAL: Must remain autouse - prevents OOM in 20K+ test suite.
    Memory buildup affects ALL tests, not just memory-intensive ones.
    """
    yield
    import gc
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        gc.collect()


# REMOVE autouse: cleanup_matplotlib (opt-in for visualization tests)
@pytest.fixture(scope="module")  # Remove autouse=True
def cleanup_matplotlib():
    """Close matplotlib figures after module (opt-in).

    OPTIMIZATION: Changed from autouse to opt-in (saves ~0.4s per run).
    Only needed by tests that create plots (~50 modules).

    Usage:
        pytestmark = pytest.mark.usefixtures("cleanup_matplotlib")
    """
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except ImportError:
        pass


# REMOVE autouse: reset_warnings_state (opt-in for tests that modify warnings)
@pytest.fixture(scope="module")  # Remove autouse=True
def reset_warnings_state():
    """Reset warnings state after module (opt-in).

    OPTIMIZATION: Changed from autouse to opt-in (saves ~3.9s per run).
    Only needed by tests that modify warnings.simplefilter() (~20 modules).

    Usage:
        pytestmark = pytest.mark.usefixtures("reset_warnings_state")
    """
    import warnings
    original_filters = warnings.filters[:]
    yield
    warnings.filters[:] = original_filters
    warnings.resetwarnings()
    import sys
    for name, module in sys.modules.items():
        if name.startswith(("oscura", "tests")) and hasattr(module, "__warningregistry__"):
            module.__warningregistry__.clear()


# REMOVE autouse: reset_logging_state (opt-in for logging tests)
@pytest.fixture(scope="module")  # Remove autouse=True
def reset_logging_state():
    """Reset logging configuration after module (opt-in).

    OPTIMIZATION: Changed from autouse to opt-in (saves ~19s per run).
    Only needed by tests that call configure_logging() (~15 modules).

    Usage:
        pytestmark = pytest.mark.usefixtures("reset_logging_state")
    """
    import logging
    original_levels = {}
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        original_levels[name] = logger.level
    yield
    for name, level in original_levels.items():
        logger = logging.getLogger(name)
        logger.setLevel(level)
```

**Migration**: Add `pytestmark` to affected test modules

```python
# Example: tests/unit/visualization/test_waveform_plot.py
import pytest

pytestmark = pytest.mark.usefixtures("cleanup_matplotlib")

def test_plot_waveform():
    """Test that uses matplotlib."""
    # ... test code
```

**Validation**:

```bash
# Measure test collection + setup time before
time uv run pytest tests/unit/utils/ --collect-only

# Apply changes, measure after (should be ~5-10s faster)
time uv run pytest tests/unit/utils/ --collect-only
```

**Completion Criteria**:

- [ ] 3 autouse fixtures converted to opt-in (keep `memory_cleanup`)
- [ ] `pytestmark` added to ~85 affected test modules
- [ ] Test execution time reduced by 5-10 seconds
- [ ] All tests still pass with new fixture strategy
- [ ] Documentation updated explaining opt-in rationale

---

### Issue 4: Test Collection Optimization

**Problem**: 12.52s collection time suggests inefficient test discovery.

**Root Causes**:

- Pytest recursively searches all directories
- Import time for large test modules
- No explicit `testpaths` optimization in CI

**Files Affected**:

- `pyproject.toml` (line 219: `testpaths = ["tests"]`)
- Test modules with heavy imports

**Fix**:

```toml
# pyproject.toml [tool.pytest.ini_options]

# Optimize test discovery (reduce collection time)
testpaths = ["tests"]  # Already present - good
norecursedirs = [
    ".*", "build", "dist", "*.egg", "node_modules", "__pycache__",
    ".venv", "venv",
    # Add subdirectories that should be skipped
    "test_data", "demos", "examples", "docs", ".claude",
]

# Import optimization
addopts = [
    "--import-mode=importlib",  # Already present - good (faster than prepend)
    "-ra",
    "--strict-markers",
    "--strict-config",
    "--tb=line",
    "--capture=sys",
    "--ff",
    "--nf",
    # Collection optimization
    "--collect-in-virtualenv",  # Collect from virtualenv (faster)
    # Disable unnecessary plugins during collection
    "-p", "no:deadfixtures",
    "-p", "no:memray",
    "-p", "no:split",
    "-p", "no:cacheprovider",  # Disable if not using --lf/--ff in CI
]
```

**Defer Heavy Imports in Test Modules**:

```python
# Example: tests/unit/analyzers/spectral/test_fft.py

# BEFORE (imports at module level - slow collection)
import numpy as np
from scipy.fft import fft, ifft
from oscura.analyzers.spectral.fft import FFTAnalyzer

# AFTER (defer heavy imports to test functions - fast collection)
import numpy as np  # Keep lightweight imports at top

def test_fft_analysis():
    """Test FFT analysis."""
    # Import heavy dependencies only when test runs
    from scipy.fft import fft, ifft
    from oscura.analyzers.spectral.fft import FFTAnalyzer

    # ... test code
```

**Validation**:

```bash
# Measure collection time before
time uv run pytest --collect-only tests/ 2>&1 | grep "collected"

# Apply changes, measure after (target: <6s)
time uv run pytest --collect-only tests/ 2>&1 | grep "collected"
```

**Completion Criteria**:

- [ ] `norecursedirs` expanded to skip non-test directories
- [ ] Heavy imports deferred in top 10 slowest test modules
- [ ] Collection time reduced from 12.52s to <6s (50% improvement)
- [ ] CI collection time monitored in future runs
- [ ] Documentation added explaining collection optimization

---

## Performance Optimizations (Month 1)

### Optimization 1: Coverage Overhead Reduction (30% → 20%)

**Current State**: Coverage adds ~30% execution overhead using default trace method.

**Target**: Reduce to ~20% using Python 3.12+ `sys.monitoring` API.

**Files Affected**:

- `pyproject.toml` ([tool.coverage.run])
- `.github/workflows/ci.yml` (env variables)
- `scripts/test.sh` (test runner script)

**Implementation**:

```bash
# .github/workflows/ci.yml (add to env section)
env:
  PYTHON_VERSION: "3.12"
  UV_CACHE_DIR: /tmp/.uv-cache
  # Coverage optimization: Use Python 3.12+ sys.monitoring API (53% faster)
  COVERAGE_CORE: sysmon  # ADD THIS LINE
  RETENTION_SHORT: 14
  RETENTION_MEDIUM: 30
  RETENTION_LONG: 90
```

```toml
# pyproject.toml [tool.coverage.run]
source = ["oscura"]
branch = true
parallel = true
data_file = ".coverage"
relative_files = true

# Python 3.12+ optimization: Use sys.monitoring API for faster coverage
# COVERAGE_CORE=sysmon reduces overhead from 30% to 20% (53% faster)
# Requires: Python 3.12+, coverage 7.4.0+
# Environment variable takes precedence, this is documentation
# Set COVERAGE_CORE=sysmon in CI/local environments

omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__pycache__/*",
]
```

```bash
# scripts/test.sh (add at top)
#!/usr/bin/env bash
set -euo pipefail

# Use fast coverage if Python 3.12+
PYTHON_VERSION=$(python --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$PYTHON_VERSION" == "3.12" ]] || [[ "$PYTHON_VERSION" == "3.13" ]]; then
    export COVERAGE_CORE=sysmon
    echo "Using fast coverage (sys.monitoring API)"
fi

# ... rest of test.sh
```

**Validation**:

```bash
# Measure test time WITHOUT coverage
time uv run pytest tests/unit/analyzers/digital/ -n 2

# Measure test time WITH old coverage
time uv run pytest tests/unit/analyzers/digital/ -n 2 --cov=oscura

# Measure test time WITH new coverage
COVERAGE_CORE=sysmon time uv run pytest tests/unit/analyzers/digital/ -n 2 --cov=oscura

# Expected: new coverage only ~20% slower than no coverage (vs 30% before)
```

**Expected Impact**:

- **Before**: 10min test → 13min with coverage (+30%)
- **After**: 10min test → 12min with coverage (+20%)
- **Savings**: 1 minute per CI run × 17 groups = 17 minutes total CI time saved

**Completion Criteria**:

- [ ] `COVERAGE_CORE=sysmon` added to CI workflows
- [ ] `scripts/test.sh` detects Python version and sets env var
- [ ] Coverage documentation updated explaining optimization
- [ ] Benchmark confirms 20% overhead (down from 30%)
- [ ] All coverage reports still accurate

---

### Optimization 2: Worker Allocation Alignment

**Current State**: CI uses 2 workers for analyzers, 4 for utils. Some groups may benefit from rebalancing.

**Analysis Needed**: Measure actual memory usage per test group to optimize worker allocation.

**Files Affected**:

- `.github/workflows/ci.yml` (lines 345-353)
- `.github/workflows/tests-chunked.yml` (lines 113-221)

**Current Worker Allocation**:

```yaml
# ci.yml - worker allocation
analyzers-*: 2 workers (memory-intensive)
non-unit-tests: 2 workers (memory-intensive)
core-protocols-loaders: 4 workers (assumed standard)
unit-*: 4 workers (standard)
```

**Investigation Strategy**:

```bash
# Add memory profiling to test runs (local only)
uv run pytest tests/unit/analyzers/digital/ -n 2 --durations=0 --verbose \
    --profile-memory > memory-profile-analyzers-digital.txt

# Analyze memory peaks per worker
grep "Peak memory" memory-profile-analyzers-digital.txt | sort -n
```

**Potential Rebalancing**:

Based on test patterns, consider:

- `core-protocols-loaders`: May need 2 workers (file I/O intensive)
- `unit-discovery-inference`: May need 2 workers (ML algorithm intensive)
- Some analyzer groups: May support 3 workers (middle ground)

**Implementation** (after profiling):

```yaml
# .github/workflows/ci.yml
- name: Run tests with coverage (${{ matrix.test-group }})
  run: |
    # Granular worker allocation based on profiling
    case "${{ matrix.test-group }}" in
      analyzers-1|analyzers-2|analyzers-3*)
        WORKERS=2  # Confirmed memory-intensive
        ;;
      core-protocols-loaders)
        WORKERS=2  # I/O-intensive (file loading)
        ;;
      unit-discovery-inference)
        WORKERS=2  # ML algorithm intensive
        ;;
      non-unit-tests)
        WORKERS=2  # Integration tests (memory-intensive)
        ;;
      *)
        WORKERS=4  # Standard tests
        ;;
    esac

    XDIST_ARGS="-n $WORKERS --maxprocesses=$WORKERS --dist loadscope"
    echo "Using $WORKERS workers for ${{ matrix.test-group }}"

    # ... run pytest
```

**Completion Criteria**:

- [ ] Memory profiling completed for all 17 test groups
- [ ] Worker allocation adjusted based on actual memory usage
- [ ] Documentation updated with profiling results
- [ ] CI runs validate new worker allocation
- [ ] No OOM errors or timeout issues

---

### Optimization 3: CI Batch Consolidation (17 groups → 12-13)

**Problem**: 17 test groups create coordination overhead. Some groups are small and can be merged.

**Current Groups** (ci.yml):

1. analyzers-1 (digital, protocols, waveform, eye, jitter)
2. analyzers-2 (spectral, power, patterns, statistical)
3. analyzers-3a (ml, side_channel)
4. analyzers-3b-parser (packet parser tests)
5. analyzers-3b-stream (packet stream tests)
6. analyzers-3b-metrics (packet metrics tests)
7. analyzers-3b-part2 (packet daq, payload tests)
8. analyzers-3e (packet hypothesis tests)
9. analyzers-3c (root analyzer tests, analysis, signal, correlation)
10. analyzers-3d (signal_integrity)
11. core-protocols-loaders
12. unit-root-tests
13. cli-ui-reporting
14. unit-workflows
15. unit-discovery-inference
16. unit-utils
17. non-unit-tests

**Analysis**: Packet analyzer split into 5 batches (3b-parser, 3b-stream, 3b-metrics, 3b-part2, 3e) suggests over-fragmentation.

**Proposed Consolidation**:

```yaml
# Merge packet analyzer batches into 2 groups instead of 5
"analyzers-3b-fast":
  # Merge: 3b-parser, 3b-stream, 3b-metrics, 3b-part2 (~4-6 min)
  paths: tests/unit/analyzers/packet/ --ignore=tests/unit/analyzers/packet/test_checksum_hypothesis.py --ignore=tests/unit/analyzers/packet/test_framing_hypothesis.py
  workers: 2

"analyzers-3b-hypothesis":
  # Keep hypothesis tests separate (slow)
  paths: tests/unit/analyzers/packet/test_checksum_hypothesis.py tests/unit/analyzers/packet/test_framing_hypothesis.py
  workers: 2

# Merge small groups
"analyzers-misc":
  # Merge: 3c (root tests), 3d (signal_integrity), 3e moved to 3b-hypothesis
  paths: tests/unit/analyzers/test_*.py tests/unit/analysis/ tests/unit/signal/ tests/unit/correlation/ tests/unit/analyzers/signal_integrity/
  workers: 2

# Merge non-unit tests
"integration-all":
  # Merge: non-unit-tests → integration, compliance, validation
  paths: tests/integration/ tests/compliance/ tests/validation/
  workers: 2
```

**New Group Structure** (13 groups instead of 17):

1. analyzers-1 (digital, protocols, waveform, eye, jitter)
2. analyzers-2 (spectral, power, patterns, statistical)
3. analyzers-3a (ml, side_channel)
4. **analyzers-3b-fast** (packet fast tests - MERGED)
5. **analyzers-3b-hypothesis** (packet hypothesis tests - MERGED)
6. **analyzers-misc** (root, signal_integrity, analysis, signal, correlation - MERGED)
7. core-protocols-loaders
8. unit-root-tests
9. cli-ui-reporting
10. unit-workflows
11. unit-discovery-inference
12. unit-utils
13. **integration-all** (integration, compliance, validation - MERGED)

**Validation**:

```bash
# Test new batch locally (should complete in <15 minutes)
time uv run pytest tests/unit/analyzers/packet/ \
    --ignore=tests/unit/analyzers/packet/test_checksum_hypothesis.py \
    --ignore=tests/unit/analyzers/packet/test_framing_hypothesis.py \
    -n 2 --dist loadscope

# Should take ~6-8 minutes (well under 25min timeout)
```

**Expected Impact**:

- **Before**: 17 groups × 2 Python versions = 34 CI jobs
- **After**: 13 groups × 2 Python versions = 26 CI jobs
- **Savings**: 8 fewer jobs = faster CI feedback (reduced queue time)
- **Coordination overhead**: Reduced from ~17min to ~13min (artifact upload/download)

**Completion Criteria**:

- [ ] Packet analyzer consolidated from 5 groups to 2
- [ ] Small groups merged where appropriate
- [ ] All consolidated groups complete in <20 minutes (80% of timeout)
- [ ] CI matrix reduced from 34 to 26 jobs
- [ ] Documentation updated with new batch structure

---

## CI/CD Restructuring (Month 1-2)

### Restructuring 1: Optimal Batch Configuration with Durations

**Goal**: Document and validate duration targets for each test group.

**Current State**: Duration warnings at 80% threshold (20min of 25min timeout) suggest some groups approaching limits.

**Target Durations**:

| Test Group | Target Duration | Max Duration | Worker Count |
|------------|----------------|--------------|--------------|
| analyzers-1 | 8-10 min | 20 min | 2 |
| analyzers-2 | 8-10 min | 20 min | 2 |
| analyzers-3a | 6-8 min | 20 min | 2 |
| analyzers-3b-fast | 6-8 min | 20 min | 2 |
| analyzers-3b-hypothesis | 10-12 min | 20 min | 2 |
| analyzers-misc | 5-7 min | 20 min | 2 |
| core-protocols-loaders | 8-10 min | 20 min | 2 |
| unit-root-tests | 3-5 min | 15 min | 4 |
| cli-ui-reporting | 5-7 min | 15 min | 4 |
| unit-workflows | 6-8 min | 15 min | 4 |
| unit-discovery-inference | 7-9 min | 20 min | 2 |
| unit-utils | 5-7 min | 15 min | 4 |
| integration-all | 8-10 min | 20 min | 2 |

**Implementation**:

```yaml
# .github/workflows/ci.yml
- name: Run tests with coverage (${{ matrix.test-group }})
  id: run-tests
  run: |
    # Duration targets for monitoring
    case "${{ matrix.test-group }}" in
      analyzers-1|analyzers-2|core-protocols-loaders|unit-discovery-inference|integration-all)
        TARGET_MIN=8
        TARGET_MAX=10
        MAX_DURATION=20
        ;;
      analyzers-3b-hypothesis)
        TARGET_MIN=10
        TARGET_MAX=12
        MAX_DURATION=20
        ;;
      analyzers-3a|analyzers-3b-fast|cli-ui-reporting|unit-workflows|unit-utils)
        TARGET_MIN=5
        TARGET_MAX=8
        MAX_DURATION=15
        ;;
      analyzers-misc|unit-root-tests)
        TARGET_MIN=3
        TARGET_MAX=7
        MAX_DURATION=15
        ;;
    esac

    echo "Target duration: ${TARGET_MIN}-${TARGET_MAX} min (max: ${MAX_DURATION} min)"

    # ... run tests

    # Check if duration exceeds target (warn if approaching max)
    if [ "$DURATION_MINUTES" -gt "$TARGET_MAX" ]; then
      echo "::warning::Test group ${{ matrix.test-group }} took ${DURATION_MINUTES}m (target: ${TARGET_MIN}-${TARGET_MAX}m)"
    fi

    # Error if exceeds max duration (leaving 5min buffer before timeout)
    if [ "$DURATION_MINUTES" -gt "$MAX_DURATION" ]; then
      echo "::error::Test group ${{ matrix.test-group }} exceeded max duration (${DURATION_MINUTES}m > ${MAX_DURATION}m)"
      exit 1
    fi
```

**Monitoring**: Add test analytics to track duration trends over time.

**Completion Criteria**:

- [ ] Duration targets documented for all 13 test groups
- [ ] CI warnings added for groups exceeding target duration
- [ ] CI errors added for groups exceeding max duration
- [ ] Historical duration data collected (3 weeks minimum)
- [ ] Groups consistently approaching limits identified for further splitting

---

### Restructuring 2: Caching Improvements

**Current Caching** (ci.yml):

- UV cache (dependencies)
- pytest cache (--ff, --lf)
- Hypothesis examples database
- mypy cache

**Missing Optimizations**:

- Numba JIT cache (would speed up analyzer tests)
- Test data cache (if synthetic data generated at runtime)

**Implementation**:

```yaml
# .github/workflows/ci.yml (add new cache step)
- name: Cache Numba JIT compilation
  uses: actions/cache@v5
  with:
    path: ~/.cache/numba
    key: numba-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('src/oscura/**/*.py') }}
    restore-keys: |
      numba-${{ runner.os }}-${{ matrix.python-version }}-
      numba-${{ runner.os }}-

- name: Cache pytest
  uses: actions/cache@v5
  with:
    path: .pytest_cache
    key: pytest-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('pyproject.toml', 'tests/**/*.py') }}
    restore-keys: |
      pytest-${{ runner.os }}-${{ matrix.python-version }}-
```

**Expected Impact**:

- Numba cache: 10-20% speedup on first-time analyzer tests (after cache hit)
- Pytest cache: Improved --ff effectiveness (failed tests run first)

**Completion Criteria**:

- [ ] Numba cache added to CI workflows
- [ ] Pytest cache key includes test file hashes
- [ ] Cache hit rate monitored (target: >80%)
- [ ] Documentation updated explaining cache strategy

---

### Restructuring 3: Timeout Threshold Adjustments

**Current State**: 25-minute job timeout with warnings at 20 minutes (80%).

**Problem**: Some groups occasionally approach 20-minute warning threshold, suggesting insufficient buffer.

**Proposed Changes**:

```yaml
# .github/workflows/ci.yml
jobs:
  test:
    timeout-minutes: 25  # Keep same (GitHub Actions max for free tier)

    steps:
      - name: Run tests with coverage
        run: |
          # Per-test timeout: 120 seconds (2 minutes)
          # Most tests complete in <1s, this catches hangs

          uv run python -m pytest ... \
            --timeout=120 \
            --timeout-method=thread \
            ...

      - name: Check test duration
        run: |
          THRESHOLD=$((25 * 60 * 70 / 100))  # 70% of timeout (17.5 min)
          CRITICAL=$((25 * 60 * 85 / 100))   # 85% of timeout (21.25 min)

          if [ "$DURATION" -gt "$CRITICAL" ]; then
            echo "::error::CRITICAL: Test approaching timeout ($DURATION_MIN min / 25 min)"
            echo "::error::This group MUST be split to prevent CI failures"
          elif [ "$DURATION" -gt "$THRESHOLD" ]; then
            echo "::warning::Test took ${DURATION_MIN}m, approaching timeout. Consider splitting."
          fi
```

**Completion Criteria**:

- [ ] Per-test timeout enforced at 120 seconds
- [ ] Job timeout warnings adjusted to 70% (17.5 min)
- [ ] Critical warnings added at 85% (21.25 min)
- [ ] Historical data confirms no groups exceed thresholds
- [ ] Documentation explains timeout rationale

---

## Long-Term Architecture (Quarter 1)

### Architecture 1: Test Organization Improvements

**Goal**: Improve test discoverability and reduce 1:1 mapping gaps (463 source files lack test files).

**Current Issues** (from test-audit-2026-01-25.md):

- 463 source files lack corresponding test files
- 35 test files in non-standard locations (automotive/, compliance/, validation/)
- 14 conftest.py files create fixture sprawl

**Proposed Structure**:

```
tests/
  ├── unit/                      # Unit tests (mirror src/ structure)
  │   ├── analyzers/
  │   ├── core/
  │   ├── loaders/
  │   ├── ...
  │   └── conftest.py            # Unit test fixtures only
  ├── integration/               # Integration tests
  │   ├── workflows/
  │   ├── end_to_end/
  │   └── conftest.py            # Integration fixtures
  ├── compliance/                # Standards compliance (IEEE, JEDEC)
  │   └── conftest.py            # Compliance fixtures
  ├── validation/                # Ground truth validation
  │   └── conftest.py            # Validation fixtures
  ├── performance/               # Performance benchmarks
  │   └── conftest.py            # Benchmark fixtures
  ├── conftest.py                # Global fixtures (64 fixtures)
  └── fixtures/                  # Shared fixture modules
      ├── signal_builders.py
      ├── packet_factories.py
      └── ...
```

**Migration Plan**:

1. **Consolidate conftest.py files** (14 → 6):
   - Keep: `conftest.py`, `unit/conftest.py`, `integration/conftest.py`, `compliance/conftest.py`, `validation/conftest.py`, `performance/conftest.py`
   - Remove: 8 subdirectory conftest files (merge fixtures into parents)

2. **Relocate non-standard test files**:
   - Move `automotive/test_*.py` → `unit/automotive/`
   - Move `validation/test_*.py` (if redundant) → `unit/validation/` or delete
   - Move `compliance/test_*.py` → Keep in `compliance/` (correct location)

3. **Create test stubs for 463 untested files**:

   ```bash
   # Script to generate test stubs
   python scripts/testing/generate_test_stubs.py

   # Creates test files with:
   # - Proper imports
   # - TODO markers for each public function
   # - Pytest markers
   ```

**Completion Criteria**:

- [ ] Conftest.py files reduced from 14 to 6
- [ ] All test files in standard locations (unit/, integration/, etc.)
- [ ] Test stub generation script created
- [ ] 463 missing test files documented with stubs
- [ ] Coverage report shows improved structure

---

### Architecture 2: Flaky Test Detection System

**Goal**: Automatically detect, quarantine, and track flaky tests.

**Implementation**:

```python
# tests/plugins/flaky_detection.py
"""Pytest plugin for automatic flaky test detection."""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

class FlakyTestDetector:
    """Detect and track flaky tests across multiple runs."""

    def __init__(self):
        self.flaky_history_file = Path(".pytest_flaky_history.json")
        self.flaky_history = self._load_history()
        self.current_run_failures = defaultdict(int)
        self.current_run_passes = defaultdict(int)

    def _load_history(self):
        """Load flaky test history from disk."""
        if self.flaky_history_file.exists():
            return json.loads(self.flaky_history_file.read_text())
        return {}

    def pytest_runtest_logreport(self, report):
        """Track test outcomes."""
        if report.when == "call":
            nodeid = report.nodeid
            if report.outcome == "passed":
                self.current_run_passes[nodeid] += 1
            elif report.outcome == "failed":
                self.current_run_failures[nodeid] += 1

    def pytest_sessionfinish(self):
        """Detect flaky tests after session."""
        flaky_tests = []

        for nodeid in self.current_run_passes:
            if nodeid in self.current_run_failures:
                # Test both passed and failed in same run (with retries)
                flaky_tests.append({
                    "nodeid": nodeid,
                    "passes": self.current_run_passes[nodeid],
                    "failures": self.current_run_failures[nodeid],
                    "detected_at": datetime.now().isoformat()
                })

        if flaky_tests:
            print(f"\n⚠️  Detected {len(flaky_tests)} flaky tests:")
            for test in flaky_tests:
                print(f"  - {test['nodeid']} (passed {test['passes']}, failed {test['failures']})")

            # Update history
            for test in flaky_tests:
                nodeid = test["nodeid"]
                if nodeid not in self.flaky_history:
                    self.flaky_history[nodeid] = []
                self.flaky_history[nodeid].append(test)

            # Save history
            self.flaky_history_file.write_text(json.dumps(self.flaky_history, indent=2))

            # Generate report
            self._generate_flaky_report()

    def _generate_flaky_report(self):
        """Generate HTML report of flaky tests."""
        report_file = Path("test-results/flaky-tests-report.html")
        report_file.parent.mkdir(exist_ok=True)

        # Calculate flakiness score (number of times test has been flaky)
        flaky_scores = {
            nodeid: len(history)
            for nodeid, history in self.flaky_history.items()
        }

        # Sort by flakiness score
        sorted_tests = sorted(flaky_scores.items(), key=lambda x: x[1], reverse=True)

        # Generate HTML report
        html = self._render_html_report(sorted_tests)
        report_file.write_text(html)
        print(f"\n📊 Flaky test report generated: {report_file}")

# Register plugin
def pytest_configure(config):
    """Register flaky detection plugin."""
    config.pluginmanager.register(FlakyTestDetector(), "flaky_detector")
```

```toml
# pyproject.toml - Enable flaky detection
[tool.pytest.ini_options]
# Enable flaky test detection plugin
plugins = ["tests.plugins.flaky_detection"]

# Quarantine tests marked as flaky
markers = [
    "flaky: Test is known to be flaky (quarantined, requires investigation)",
    # ... other markers
]

# Skip quarantined tests by default
addopts = [
    "-m", "not flaky",  # Skip flaky tests unless explicitly requested
    # ... other options
]
```

**Usage**:

```bash
# Run tests with flaky detection
uv run pytest tests/ --reruns=2

# Review flaky test report
open test-results/flaky-tests-report.html

# Run only flaky tests for debugging
uv run pytest tests/ -m flaky
```

**Completion Criteria**:

- [ ] Flaky detection plugin implemented
- [ ] HTML report generation working
- [ ] Flaky tests automatically quarantined
- [ ] CI integration (upload flaky report as artifact)
- [ ] Documentation explaining quarantine process

---

### Architecture 3: Test Analytics Pipeline

**Goal**: Collect and analyze test execution metrics over time to identify trends.

**Metrics to Track**:

- Test duration per group (detect slowdown)
- Test failure rate (detect instability)
- Flaky test rate (detect quality degradation)
- Coverage percentage (detect gaps)
- Memory usage per group (detect leaks)
- Collection time (detect import bloat)

**Implementation**:

```yaml
# .github/workflows/test-analytics.yml
name: Test Analytics

on:
  workflow_run:
    workflows: ["CI", "Test Suite (Chunked)"]
    types: [completed]

jobs:
  collect-metrics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6

      - name: Download test results
        uses: actions/download-artifact@v7
        with:
          name: test-results-*
          path: test-results/

      - name: Parse test results
        run: |
          python scripts/analytics/parse_test_results.py \
            --input test-results/ \
            --output analytics/test-metrics.json

      - name: Upload to analytics database
        run: |
          python scripts/analytics/upload_metrics.py \
            --metrics analytics/test-metrics.json \
            --database ${{ secrets.ANALYTICS_DB_URL }}

      - name: Generate trend report
        run: |
          python scripts/analytics/generate_trend_report.py \
            --output analytics/trend-report.html

      - name: Upload trend report
        uses: actions/upload-artifact@v6
        with:
          name: test-analytics-report
          path: analytics/trend-report.html
          retention-days: 90
```

```python
# scripts/analytics/parse_test_results.py
"""Parse JUnit XML test results and extract metrics."""

import xml.etree.ElementTree as ET
from pathlib import Path
import json
from datetime import datetime

def parse_junit_xml(xml_file: Path) -> dict:
    """Extract metrics from JUnit XML file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    testsuite = root.find("testsuite")

    return {
        "file": xml_file.name,
        "timestamp": datetime.now().isoformat(),
        "tests": int(testsuite.get("tests", 0)),
        "failures": int(testsuite.get("failures", 0)),
        "errors": int(testsuite.get("errors", 0)),
        "skipped": int(testsuite.get("skipped", 0)),
        "time": float(testsuite.get("time", 0.0)),
        "test_cases": [
            {
                "name": testcase.get("name"),
                "classname": testcase.get("classname"),
                "time": float(testcase.get("time", 0.0)),
                "status": "passed" if testcase.find("failure") is None else "failed"
            }
            for testcase in testsuite.findall("testcase")
        ]
    }

def main():
    """Parse all test result files."""
    results_dir = Path("test-results")
    metrics = []

    for xml_file in results_dir.glob("*.xml"):
        metrics.append(parse_junit_xml(xml_file))

    # Write aggregated metrics
    output_file = Path("analytics/test-metrics.json")
    output_file.parent.mkdir(exist_ok=True)
    output_file.write_text(json.dumps(metrics, indent=2))

    print(f"Parsed {len(metrics)} test result files")

if __name__ == "__main__":
    main()
```

**Dashboard**: Use Grafana or similar to visualize trends.

**Completion Criteria**:

- [ ] Test analytics workflow created
- [ ] Metric parsing scripts implemented
- [ ] Analytics database configured (SQLite for MVP)
- [ ] Trend report generation working
- [ ] Dashboard deployed for team access

---

### Architecture 4: Continuous Optimization Process

**Goal**: Establish regular review cycles to keep test suite optimized.

**Process**:

1. **Weekly Review** (automated):
   - Check for test groups approaching timeout thresholds
   - Identify new flaky tests
   - Review test failure trends

2. **Monthly Review** (manual):
   - Analyze test duration trends (are tests getting slower?)
   - Review coverage gaps (new code missing tests?)
   - Evaluate worker allocation (memory profiling)
   - Audit autouse fixtures (still necessary?)

3. **Quarterly Review** (strategic):
   - Major refactoring of slow test groups
   - Consolidate/split test batches as needed
   - Update test architecture based on learnings
   - Review and update this implementation plan

**Automation**:

```yaml
# .github/workflows/test-health-check.yml
name: Test Suite Health Check

on:
  schedule:
    - cron: "0 9 * * 1"  # Every Monday at 9 AM

jobs:
  health-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6

      - name: Generate health report
        run: |
          python scripts/analytics/test_health_check.py \
            --lookback-days 7 \
            --output health-report.md

      - name: Create issue if problems detected
        uses: actions/github-script@v8
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('health-report.md', 'utf8');

            // Parse report for issues
            const issues = report.match(/⚠️ WARNING:.*/g) || [];
            const criticals = report.match(/🚨 CRITICAL:.*/g) || [];

            if (criticals.length > 0 || issues.length > 3) {
              await github.rest.issues.create({
                owner: context.repo.owner,
                repo: context.repo.repo,
                title: '[Test Health] Issues detected in test suite',
                body: report,
                labels: ['testing', 'maintenance']
              });
            }
```

**Completion Criteria**:

- [ ] Weekly health check workflow created
- [ ] Health report script implemented
- [ ] Automatic issue creation for critical problems
- [ ] Monthly review process documented
- [ ] Quarterly review scheduled in team calendar

---

## Implementation Checklist

### Week 1: Critical Fixes

- [ ] **Day 1-2**: Fix pytest-timeout configuration (Issue 1)
  - Uncomment timeout settings in pyproject.toml
  - Add pytest-rerunfailures to dependencies
  - Validate locally
- [ ] **Day 2-3**: Standardize flaky test retry (Issue 2)
  - Add retry config to pyproject.toml addopts
  - Update CI workflows to inherit config
  - Test with intentionally flaky test
- [ ] **Day 3-4**: Optimize autouse fixtures (Issue 3)
  - Convert 3 fixtures to opt-in
  - Add pytestmark to affected test modules (~85 files)
  - Measure performance improvement
- [ ] **Day 4-5**: Optimize test collection (Issue 4)
  - Expand norecursedirs in pyproject.toml
  - Defer heavy imports in top 10 slow modules
  - Benchmark collection time improvement

**Success Metrics**:

- Test collection: 12.52s → <6s (-50%)
- Fixture overhead: ~27s → ~10s (-62%)
- Configuration: All timeouts consistent
- All tests passing with new configuration

### Week 2-4: Performance Optimizations

- [ ] **Week 2**: Coverage optimization (Optimization 1)
  - Add COVERAGE_CORE=sysmon to CI
  - Update scripts/test.sh with auto-detection
  - Benchmark coverage overhead improvement
- [ ] **Week 2**: Worker allocation profiling (Optimization 2)
  - Profile memory usage for all 17 test groups
  - Adjust worker counts based on data
  - Validate no OOM errors
- [ ] **Week 3**: CI batch consolidation (Optimization 3)
  - Merge packet analyzer from 5 to 2 groups
  - Merge small groups where appropriate
  - Test all consolidated groups locally
- [ ] **Week 4**: Validate all optimizations
  - Full CI run with all optimizations enabled
  - Compare before/after metrics
  - Document improvements

**Success Metrics**:

- Coverage overhead: 30% → 20% (-33% relative)
- CI groups: 17 → 13 (-24%)
- CI job matrix: 34 → 26 (-24%)
- Total CI time: Measure improvement

### Month 2: CI/CD Restructuring

- [ ] **Week 5**: Optimal batch configuration (Restructuring 1)
  - Document duration targets for all groups
  - Add duration warnings to CI
  - Collect 3 weeks of duration data
- [ ] **Week 6**: Caching improvements (Restructuring 2)
  - Add Numba cache to workflows
  - Improve pytest cache key specificity
  - Monitor cache hit rates
- [ ] **Week 7**: Timeout adjustments (Restructuring 3)
  - Implement per-test timeout (120s)
  - Adjust job timeout warnings (70%, 85%)
  - Validate no false positives
- [ ] **Week 8**: Validation and documentation
  - Complete CI/CD documentation
  - Create runbook for timeout issues
  - Train team on new structure

**Success Metrics**:

- All test groups complete within target durations
- Cache hit rate: >80%
- Zero timeout false positives
- Complete documentation

### Quarter 1: Long-Term Architecture

- [ ] **Month 2-3**: Test organization (Architecture 1)
  - Consolidate conftest.py files (14 → 6)
  - Relocate non-standard test files
  - Create test stub generation script
  - Generate stubs for 463 untested files
- [ ] **Month 3**: Flaky test detection (Architecture 2)
  - Implement flaky detection plugin
  - Add quarantine system
  - Create HTML report generation
  - Integrate with CI
- [ ] **Month 3**: Test analytics (Architecture 3)
  - Create test analytics workflow
  - Implement metric parsing scripts
  - Set up analytics database
  - Deploy dashboard
- [ ] **Ongoing**: Continuous optimization (Architecture 4)
  - Implement weekly health check workflow
  - Schedule monthly manual reviews
  - Schedule quarterly strategic reviews
  - Document learnings and update plan

**Success Metrics**:

- Conftest files: 14 → 6 (-57%)
- Test stubs created: 463 files
- Flaky tests detected and quarantined: >90% accuracy
- Analytics dashboard live
- Optimization process documented

---

## Expected Outcomes

### Performance Improvements (Quantified)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Test Collection Time** | 12.52s | ~5s | -60% (7.5s saved) |
| **Autouse Fixture Overhead** | ~27s | ~10s | -63% (17s saved) |
| **Coverage Overhead** | +30% | +20% | -33% relative |
| **Per-Group CI Time** | 8-10 min | 7-9 min | -10% to -15% |
| **Total CI Time** (17→13 groups) | ~170 min | ~130 min | -24% (40 min saved) |
| **CI Job Matrix** | 34 jobs | 26 jobs | -24% (8 fewer jobs) |
| **Collection + Setup** | ~40s | ~20s | -50% (20s saved) |

**Total Time Savings per CI Run**: ~60-80 minutes (35-47% improvement)

### Reliability Improvements

| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Timeout Enforcement** | Inconsistent | Consistent (120s/test, 25min/job) | Prevents hangs |
| **Flaky Test Handling** | Manual detection | Automatic detection + quarantine | >90% accuracy |
| **Configuration Consistency** | Varies across workflows | Single source of truth (pyproject.toml) | Zero drift |
| **Batch Balance** | Some approaching timeout | All <70% of timeout | 30% safety margin |
| **Worker Allocation** | Rule-based | Data-driven (profiled) | Optimal performance |

### Maintenance Burden Reduction

| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Configuration Files** | pyproject.toml + 14 conftest | pyproject.toml + 6 conftest | -57% files |
| **CI Coordination Overhead** | 17 groups × upload/download | 13 groups × upload/download | -24% overhead |
| **Flaky Test Investigation** | Manual tracking | Automated with history | 80% time saved |
| **Test Gap Identification** | Manual code review | Automated stub generation | 95% time saved |
| **Duration Monitoring** | Manual log parsing | Automated warnings | 100% coverage |

---

## Rollback Procedures

### Rollback Plan: If Optimizations Cause Issues

**Issue**: Tests fail unexpectedly after implementing optimizations

**Rollback Steps**:

1. **Immediate**: Revert problematic commit

   ```bash
   git revert <commit-hash>
   git push origin <branch>
   ```

2. **Configuration rollback**: Comment out new config

   ```toml
   # pyproject.toml - Rollback to safe defaults
   # timeout = 120  # Commented out - use CI default
   # --reruns=2  # Commented out - no automatic retry
   ```

3. **CI rollback**: Restore previous batch structure

   ```bash
   git checkout main -- .github/workflows/ci.yml
   git commit -m "rollback: Restore CI batch structure"
   git push
   ```

4. **Incremental re-application**: Apply optimizations one at a time
   - Week 1: Only timeout + retry standardization
   - Week 2: Only autouse fixture optimization
   - Week 3: Only collection optimization
   - Week 4: Only coverage optimization

**Validation after rollback**:

```bash
# Run full test suite locally
./scripts/test.sh

# Verify CI passes
git push && watch -n 30 'gh run list --branch <branch> --limit 1'
```

---

## Appendix A: Configuration Comparison

### Before (Current)

```toml
# pyproject.toml [tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = [
    "--import-mode=importlib",
    "-ra",
    "--strict-markers",
    "--strict-config",
    "--tb=line",
    "--capture=sys",
    "--ff",
    "--nf",
    "-p", "no:deadfixtures",
    "-p", "no:memray",
    "-p", "no:split",
]
# NOTE: timeout settings commented out
# timeout = 60
# timeout_method = "thread"
```

### After (Optimized)

```toml
# pyproject.toml [tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = [
    "--import-mode=importlib",
    "-ra",
    "--strict-markers",
    "--strict-config",
    "--tb=line",
    "--capture=sys",
    "--ff",
    "--nf",
    # Flaky test handling
    "--reruns=2",
    "--reruns-delay=1",
    # Collection optimization
    "--collect-in-virtualenv",
    # Disable unnecessary plugins
    "-p", "no:deadfixtures",
    "-p", "no:memray",
    "-p", "no:split",
    "-p", "no:cacheprovider",
]
# Timeout enforcement (prevents test hangs)
timeout = 120
timeout_method = "thread"
timeout_func_only = true

# Expanded to skip non-test directories
norecursedirs = [
    ".*", "build", "dist", "*.egg", "node_modules", "__pycache__",
    ".venv", "venv",
    "test_data", "demos", "examples", "docs", ".claude",
]
```

---

## Appendix B: Test Group Duration Analysis

**Data Collection Method**:

```bash
# Extract duration from CI logs
gh run view <run-id> --log | grep "Test execution took" | sort -n
```

**Current Durations** (estimated from CI workflow):

| Group | Estimated Duration | Worker Count | Status |
|-------|-------------------|--------------|--------|
| analyzers-1 | 8-10 min | 2 | ✅ Good |
| analyzers-2 | 8-10 min | 2 | ✅ Good |
| analyzers-3a | 6-8 min | 2 | ✅ Good |
| analyzers-3b-parser | 3-4 min | 2 | ⚠️ Can merge |
| analyzers-3b-stream | 3-4 min | 2 | ⚠️ Can merge |
| analyzers-3b-metrics | 3-4 min | 2 | ⚠️ Can merge |
| analyzers-3b-part2 | 3-4 min | 2 | ⚠️ Can merge |
| analyzers-3e | 10-12 min | 2 | ✅ Good (hypothesis) |
| analyzers-3c | 5-7 min | 2 | ✅ Good |
| analyzers-3d | 4-5 min | 2 | ⚠️ Can merge with 3c |
| core-protocols-loaders | 8-10 min | 4 | ⚠️ May need 2 workers |
| unit-root-tests | 3-5 min | 4 | ✅ Good |
| cli-ui-reporting | 5-7 min | 4 | ✅ Good |
| unit-workflows | 6-8 min | 4 | ✅ Good |
| unit-discovery-inference | 7-9 min | 4 | ⚠️ May need 2 workers |
| unit-utils | 5-7 min | 4 | ✅ Good |
| non-unit-tests | 8-10 min | 2 | ✅ Good |

**Legend**:

- ✅ Good: Duration optimal, no changes needed
- ⚠️ Can merge: Duration too short, merge with another group
- ⚠️ May need adjustment: Worker count may need profiling

---

## Appendix C: References

### Best Practices Documents

- `/home/lair-click-bats/development/oscura/docs/research/pytest-large-test-suites-best-practices.md`
- Trail of Bits: Making PyPI's Test Suite 81% Faster
- pytest-with-eric: 13 Proven Ways to Improve Test Runtime

### Project Analysis Documents

- `/home/lair-click-bats/development/oscura/.claude/agent-outputs/test-audit-2026-01-25.md`
- `/home/lair-click-bats/development/oscura/docs/testing/comprehensive-test-suite-2025-01-25.md`

### Configuration Files

- `/home/lair-click-bats/development/oscura/pyproject.toml`
- `/home/lair-click-bats/development/oscura/tests/conftest.py`
- `/home/lair-click-bats/development/oscura/.github/workflows/ci.yml`
- `/home/lair-click-bats/development/oscura/.github/workflows/tests-chunked.yml`

### pytest Documentation

- [pytest Configuration Reference](https://docs.pytest.org/en/stable/reference/customize.html)
- [pytest-xdist Distribution Modes](https://pytest-xdist.readthedocs.io/en/stable/distribution.html)
- [pytest Fixture Scopes](https://docs.pytest.org/en/stable/how-to/fixtures.html#fixture-scopes)
- [Coverage.py sys.monitoring API](https://coverage.readthedocs.io/en/latest/config.html#run-core)

---

**Document Status**: Ready for Implementation
**Next Steps**: Review with team → Approve → Begin Week 1 implementation
**Owner**: Development Team
**Reviewers**: CI/CD Lead, QA Lead, Tech Lead
