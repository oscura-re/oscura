# Contributing to Oscura

First off, thank you for considering contributing to Oscura! It's people like you that make Oscura such a great tool.

## Table of Contents

- [Documentation Philosophy](#documentation-philosophy)
- [Quick Start](#quick-start)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Testing Standards](#testing-standards)
- [Code Quality Standards](#code-quality-standards)
- [Performance Patterns](#performance-patterns)
- [Security Best Practices](#security-best-practices)
- [Documentation Requirements](#documentation-requirements)
- [Git Workflow](#git-workflow)
- [IEEE Compliance Guidelines](#ieee-compliance-guidelines)
- [Troubleshooting CI Failures](#troubleshooting-ci-failures)

---

## Documentation Philosophy

Oscura follows a "demos as documentation" approach:

- **Learn by Example**: All capabilities are demonstrated in [demos/](demos/) with working code
- **Demo READMEs**: Each demo category has a comprehensive README explaining concepts
- **API Reference**: Generated documentation at [docs/api/](docs/api/)

When adding new capabilities:

1. Implement the feature
2. Add working demo in appropriate category
3. Update demo README to explain the capability
4. Ensure docstrings are complete (for API docs generation)

Do NOT create separate user guides or tutorials - they will drift out of sync. Demos are the source of truth.

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Node.js 18+ (for markdownlint)

### Complete Setup (Recommended)

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/oscura.git
cd oscura

# Complete setup (RECOMMENDED - does everything)
./scripts/setup.sh

# Verify setup
./scripts/verify-setup.sh
```

### Manual Setup (Alternative)

```bash
# Install all dependencies
uv sync --all-extras

# Install git hooks (REQUIRED - prevents CI failures)
./scripts/setup/install-hooks.sh

# Verify setup
uv run pytest tests/unit -x --maxfail=5
```

**IMPORTANT:** The `install-hooks.sh` script installs BOTH pre-commit and pre-push hooks. These hooks are **REQUIRED** to prevent CI failures. Do not skip this step!

---

## Development Workflow

### Quick Reference

```bash
# 1. Make your changes

# 2. Quick verification during development
./scripts/setup/verify.sh         # Fast: lint + format check
./scripts/setup/verify.sh --fix   # Auto-fix issues

# 3. Run comprehensive checks
./scripts/check.sh                # Lint + typecheck
./scripts/fix.sh                  # Auto-fix all issues
./scripts/test.sh                 # Run tests (optimal config)
./scripts/test.sh --fast          # Quick tests without coverage

# 4. Before pushing - full CI verification
./scripts/pre-push.sh             # Full CI verification (10-15 min)
./scripts/pre-push.sh --quick     # Fast mode (2 min)

# 5. Commit and push
git add .
git commit -m "feat: your change"  # Pre-commit hook runs automatically
git push                           # Pre-push hook runs automatically
```

### Verification Scripts

| Script | Purpose | Duration |
|---|---|---|
| `./scripts/setup/verify.sh` | Quick lint/format check | ~30s |
| `./scripts/check.sh` | Lint + typecheck | ~1m |
| `./scripts/fix.sh` | Auto-fix all issues | ~30s |
| `./scripts/test.sh` | Optimal test execution | ~8-10m |
| `./scripts/test.sh --fast` | Quick tests (no coverage) | ~5-7m |
| `./scripts/pre-push.sh` | Full CI verification | ~10-15m |

**Always use these validated scripts instead of manual commands.** They provide optimal configuration and are battle-tested through CI/CD.

### What Pre-Push Verifies

The `pre-push.sh` script mirrors the GitHub Actions CI pipeline:

**Stage 1 - Fast Checks:**

- Pre-commit hooks (ruff, format, yaml, markdown, etc.)
- Ruff lint and format verification
- MyPy type checking
- Config validation (SSOT, orchestration)

**Stage 2 - Tests:**

- Test marker validation
- Unit tests (parallelized)
- Integration tests
- Compliance tests

**Stage 3 - Build Verification:**

- MkDocs documentation build (--strict)
- Package build (uv build)
- CLI command verification
- Docstring coverage check

### Git Hooks

Oscura uses two types of git hooks to prevent CI failures:

1. **Pre-commit hooks** (via pre-commit framework) - Run quality checks on every commit
2. **Pre-push hooks** (custom) - Run comprehensive CI verification before push

#### Bypassing Git Hooks (Use Sparingly)

In rare cases, you may need to bypass git hooks:

```bash
git commit --no-verify    # Skip pre-commit hooks
git push --no-verify      # Skip pre-push hook
```

**When to bypass:**

✅ **Acceptable reasons:**

- Creating a WIP (work-in-progress) commit on a feature branch
- Emergency hotfix needed immediately (fix CI in next commit)
- Hook has a bug preventing legitimate work
- Rebasing/amending commits (hooks already ran before)

❌ **NOT acceptable:**

- "Hooks are too slow" (use `--quick` mode instead)
- "I'll fix it later" (fix it now before committing)
- Pushing to main/develop (hooks are there to protect these branches)
- Avoiding test failures (tests exist for a reason)

**Important notes:**

- **Branch protection still applies**: Even with `--no-verify`, failing code CANNOT merge to main
- **You're not circumventing CI**: GitHub CI will still run all checks
- **You're bypassing local validation**: This means pushing untested code, which will fail CI
- **Use pre-push `--quick` instead**: For faster feedback during development

**Better alternatives:**

```bash
# Instead of --no-verify, use quick mode:
./scripts/pre-push.sh --quick        # Fast checks (2 min)

# Or auto-fix issues first:
./scripts/pre-push.sh --fix          # Auto-fix then verify

# For feature branches, hooks run quick mode automatically
git push  # Quick verification for feature branches
```

---

## Testing Standards

### Running Tests

For comprehensive test documentation, see **[docs/testing/test-suite-guide.md](docs/testing/test-suite-guide.md)**.

**Recommended:** Use the optimized test script:

```bash
./scripts/test.sh              # Full tests with coverage (8-10 min)
./scripts/test.sh --fast       # Quick tests without coverage (5-7 min)
```

**Manual test commands** (only if needed for specific scenarios):

```bash
# Run unit tests
uv run pytest tests/unit -v --timeout=90

# Run tests with coverage
uv run pytest tests/unit --cov=src/oscura --cov-report=term-missing

# Run specific module tests
uv run pytest tests/unit/analyzers -v
uv run pytest tests/unit/protocols -v

# Run in parallel
uv run pytest tests/unit -n auto
```

### Writing Strong Tests

**CRITICAL**: Tests must validate behavior, not just "doesn't crash". Follow these patterns:

#### ✅ GOOD: Tests with Meaningful Assertions

```python
def test_decode_standard_frame(can_decoder):
    """Test that decoder returns valid CAN frames with correct attributes."""
    frames = can_decoder.decode(can_data)

    # Validate structure and contents
    assert len(frames) > 0, "Should decode at least one frame"
    assert frames[0].arbitration_id == 0x123
    assert len(frames[0].data) == 8
    assert frames[0].is_extended_id is False
```

#### ❌ BAD: Tests Without Assertions

```python
def test_decode_standard_frame(can_decoder):
    """Test decoder doesn't crash."""
    frames = can_decoder.decode(can_data)  # No validation!
```

#### ✅ GOOD: Exception Testing with pytest.raises

```python
def test_invalid_input_raises_error():
    """Test that decoder raises ValueError on malformed input."""
    with pytest.raises(ValueError, match="invalid CAN frame"):
        decoder.decode(malformed_data)
```

#### ❌ BAD: Exception Swallowing

```python
def test_invalid_input_raises_error():
    """Test error handling."""
    try:
        decoder.decode(malformed_data)
    except Exception:
        pass  # Test passes even if wrong exception raised!
```

#### ✅ GOOD: Visualization Testing

```python
def test_plot_tie_histogram(tie_data):
    """Test TIE histogram generates figure with data."""
    fig, ax = plot_tie(tie_data)

    assert fig is not None
    assert ax is not None
    assert len(ax.lines) > 0  # Has data plotted
    assert ax.get_xlabel()  # Has axis labels
```

#### ❌ BAD: Visualization Without Validation

```python
def test_plot_tie_histogram(tie_data):
    """Test plotting doesn't crash."""
    plot_tie(tie_data)  # No validation of plot output!
```

### Test Skip Documentation Standards

**CRITICAL**: All conditional skips MUST be documented with clear inline comments explaining WHY they skip.

Oscura has **133 valid conditional skips** (100% documented). See **[tests/SKIP_DOCUMENTATION.md](tests/SKIP_DOCUMENTATION.md)** for complete inventory.

#### Valid Skip Categories

Valid skips fall into these categories:

1. **Optional Dependencies** (97 skips) - Tests that require optional libraries
2. **Platform-Specific** (6 skips) - Tests that only run on certain platforms
3. **Test Data** (30 skips) - Tests requiring specific data files

#### ✅ GOOD: Documented Conditional Skip

```python
def test_hdf5_export(signal):
    """Test exporting signal to HDF5 format."""
    try:
        import h5py
    except ImportError:
        # SKIP: Valid - Optional h5py dependency
        # Only skip if h5py not installed (pip install oscura[hdf5])
        pytest.skip("h5py not installed")

    # Test code using h5py
    export_to_hdf5(signal, "output.h5")
```

**Documentation requirements:**

- Line 1: `# SKIP: Valid - <category>`
- Line 2: `# <explanation of when/why skip occurs>`
- Skip call: `pytest.skip("<actionable reason>")`

#### ❌ BAD: Undocumented Skip

```python
def test_hdf5_export(signal):
    """Test HDF5 export."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not installed")  # ❌ No documentation
```

#### ❌ BAD: Unconditional Skip

```python
@pytest.mark.skip("TODO: implement later")  # ❌ Always skips
def test_future_feature():
    pass
```

**Instead use xfail:**

```python
@pytest.mark.xfail(reason="Feature not yet implemented", strict=False)
def test_future_feature():
    pass
```

#### Common Skip Templates

**Optional dependency:**

```python
try:
    import matplotlib  # noqa: F401
except ImportError:
    # SKIP: Valid - Optional matplotlib dependency
    # Only skip if matplotlib not installed (pip install oscura[viz])
    pytest.skip("matplotlib not available")
```

**Platform-specific:**

```python
try:
    symlink_path.symlink_to(target_path)
except (OSError, NotImplementedError):
    # SKIP: Valid - Platform-specific test
    # Only skip on platforms without symlink support (e.g., Windows FAT32)
    pytest.skip("Symlinks not supported on this system")
```

**Test data:**

```python
wfm_files = list(test_data_dir.glob("*.wfm"))
if not wfm_files:
    # SKIP: Valid - Test data dependency
    # Only skip if Tektronix WFM test files not available
    pytest.skip("No WFM files available")
```

#### Skip Best Practices

**DO:**

- ✅ Skip when optional dependency is NOT installed
- ✅ Skip when platform lacks required feature
- ✅ Skip when test data is unavailable (document how to get it)
- ✅ Provide actionable skip reasons (tell user what to install)
- ✅ Document EVERY conditional skip with `# SKIP: Valid` comment

**DON'T:**

- ❌ Skip tests for installed dependencies
- ❌ Skip tests with "TODO" reasons (use `@pytest.mark.xfail` instead)
- ❌ Skip tests permanently without explanation
- ❌ Skip tests that could be fixed by generating test data
- ❌ Skip tests because they're flaky (fix the test!)

#### Validating Skips

Before committing, verify your skips are properly documented:

```bash
# Check for undocumented skips
python3 .claude/comprehensive_skip_documentation.py

# Verify skip patterns
grep -r "pytest.skip" tests --include="*.py" | grep -v "# SKIP: Valid"
```

All conditional skips MUST have inline documentation. Undocumented skips will be flagged in code review.

### Test Skip Patterns

All `pytest.skip()` calls in tests MUST be documented with inline comments. See [tests/SKIP_PATTERNS.md](tests/SKIP_PATTERNS.md) for comprehensive documentation.

#### ✅ Valid Conditional Skips (Allowed)

```python
try:
    import h5py
except ImportError:
    # SKIP: Valid - Optional h5py dependency
    # Only skip if h5py not installed (pip install oscura[hdf5])
    pytest.skip("h5py not installed")
```

#### ❌ Invalid Skips (Not Allowed)

```python
# ✗ INVALID - Remove or complete test
pytest.skip("TODO: implement later")
```

#### Documentation Requirements

1. All conditional skips MUST have `# SKIP: Valid - <category>` comment
2. Skip reason MUST be actionable (tell user what to install/configure)
3. No TODO/WIP skip reasons allowed
4. Reference pip extras for optional dependencies

See [tests/SKIP_PATTERNS.md](tests/SKIP_PATTERNS.md) for complete patterns and examples.

### Property-Based Testing with Hypothesis

Use `hypothesis.assume()` for preconditions, NOT `pytest.skip()`:

#### ✅ GOOD: Hypothesis assume() Pattern

```python
from hypothesis import given, assume, strategies as st

@given(samples=st.lists(st.floats()))
def test_analysis_requires_data(samples):
    """Test that analysis works with non-empty data."""
    assume(len(samples) > 0)  # Generate new data if empty
    assume(all(not math.isnan(s) for s in samples))  # No NaN values

    result = analyze(samples)
    assert result is not None
```

#### ❌ BAD: Using pytest.skip() in Hypothesis

```python
@given(samples=st.lists(st.floats()))
def test_analysis_requires_data(samples):
    """Test analysis (skips on empty data)."""
    if len(samples) == 0:
        pytest.skip("Empty samples")  # WRONG! Use assume()

    result = analyze(samples)
    assert result is not None
```

**Why assume() is correct:**

- `pytest.skip()` reports test as "skipped" (looks like coverage gap)
- `assume()` tells Hypothesis to generate new data meeting requirements
- Hypothesis tests validate invariants across FULL input space
- All generated examples are tested (not skipped)

### Test Fixtures

- MUST use fixtures from `tests/conftest.py`
- Use synthetic data only (<100KB)
- ALWAYS run via `./scripts/test.sh` (SSOT)
- DO NOT run `pytest` directly (wrong config)

### Test Markers

```python
@pytest.mark.unit              # Unit tests (fast, isolated)
@pytest.mark.integration       # Integration tests (slower)
@pytest.mark.performance       # Performance benchmarks
@pytest.mark.slow              # Tests taking >1 second
```

See [docs/testing/test-suite-guide.md](docs/testing/test-suite-guide.md) for complete marker list.

---

## Code Quality Standards

### Style Guide

- Follow PEP 8 (enforced by ruff)
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Keep functions focused and small

### Naming Conventions

From `.claude/coding-standards.yaml`:

- **Files/functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `SCREAMING_SNAKE_CASE`
- **Private**: `_leading_underscore`
- **Line length**: 100 characters

### Type Hints

**REQUIRED**: All code must pass `mypy --strict`:

```python
def process_signal(
    trace: TraceData,
    threshold: float = 0.5,
    *,
    normalize: bool = True
) -> dict[str, float]:
    """Process signal with threshold detection.

    Args:
        trace: Input waveform trace.
        threshold: Detection threshold (0-1).
        normalize: Whether to normalize output.

    Returns:
        Dictionary of measurement results.
    """
    ...
```

**Type Hint Patterns:**

```python
# NumPy arrays
from numpy.typing import NDArray
import numpy as np

def analyze(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Analyze numerical data."""
    ...

# Generic types
from typing import Any

def load_config() -> dict[str, Any]:
    """Load configuration with dynamic values."""
    ...

# Union types
from typing import Union

def parse_value(val: Union[float, np.floating[Any]]) -> float:
    """Parse numeric value from multiple types."""
    return float(val)
```

### Linting

**REQUIRED**: Code must pass `ruff check` with zero errors:

```bash
# Check specific file
uv run ruff check <file>

# Auto-fix issues
uv run ruff check --fix <file>

# Format code
uv run ruff format <file>
```

**Common Patterns:**

```python
# ✅ GOOD: isinstance with tuple
if isinstance(value, (int, float, np.number)):
    ...

# ❌ BAD: Multiple or'd isinstance calls
if isinstance(value, int) or isinstance(value, float):
    ...

# ✅ GOOD: Explicit boolean check
assert condition
assert not failed

# ❌ BAD: Comparing to True/False
assert condition == True
assert failed == False

# ✅ GOOD: Sorted __all__ exports
__all__ = [
    "AnalyzerBase",
    "SignalProcessor",
    "WaveformAnalyzer",
]

# ✅ GOOD: Efficient dict iteration
for value in my_dict.values():  # When key unused
    process(value)

for key in my_dict:  # When only key needed
    lookup(key)
```

---

## Performance Patterns

### Lazy Imports

For expensive imports that aren't always needed:

```python
# Module-level cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from expensive.module import ExpensiveClass

_impl: type[ExpensiveClass] | None = None

def expensive_function(args):
    """Function using lazy-loaded module."""
    global _impl
    if _impl is None:
        from expensive.module import ExpensiveClass
        _impl = ExpensiveClass
    return _impl(args)
```

**Benefits:**

- Import time: 14.5s → <0.1s (145x faster in real case)
- Full backward compatibility
- Type hints preserved via TYPE_CHECKING

### Caching

Use `functools.lru_cache` for expensive computations:

```python
from functools import lru_cache

@lru_cache(maxsize=32)
def _get_window_cached(window_name: str, size: int):
    """Cache window function computation."""
    return signal.get_window(window_name, size)

def compute_spectrum(data, window='hann'):
    """Compute spectrum with cached window."""
    window_arr = _get_window_cached(window, len(data))
    return np.fft.rfft(data * window_arr)
```

**Impact:** 100-1000x faster for repeated calls (10ms → 0.01ms)

### NumPy Vectorization

Replace Python loops with NumPy operations:

#### ❌ BAD: Nested Loops

```python
# O(n) loop - slow
result = 0
for i in range(len(bits)):
    result += bits[i] << i
```

#### ✅ GOOD: NumPy Vectorization

```python
# Vectorized - 10-100x faster
shifts = np.arange(len(bits))
result = np.sum(bits << shifts)
```

### Numba JIT Compilation

For numerical hot paths:

```python
from oscura.core.numba_backend import njit, prange

@njit(parallel=True, cache=True)
def _find_edges_numba(data, threshold, hysteresis):
    """JIT-compiled edge detection."""
    edges = []
    state = data[0] > threshold

    for i in prange(1, len(data)):
        if state and data[i] < threshold - hysteresis:
            edges.append((i, False))  # Falling edge
            state = False
        elif not state and data[i] > threshold + hysteresis:
            edges.append((i, True))  # Rising edge
            state = True

    return edges
```

**Impact:** 15-30x speedup on numerical loops

**Patterns:**

- `@njit(cache=True)` for single-threaded
- `@njit(parallel=True, cache=True)` for parallel loops
- Use `prange()` instead of `range()` for parallel
- Graceful fallback when Numba unavailable

### File I/O Optimization

#### Buffering

```python
# Add buffering for large files
with open(file_path, "rb", buffering=65536) as f:  # 64KB buffer
    data = f.read()
```

**Impact:** 5-15% faster for files >1MB

#### Memory-Mapped Files

```python
import mmap

# For very large files (>100MB)
with open(file_path, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    # OS handles paging automatically
    segment = np.frombuffer(mm[offset:offset+size], dtype=dtype)
```

**Impact:** 5-10x faster for multi-GB files

### Regex Optimization

#### Precompile at Module Level

```python
# At module level (NOT in function)
_TIMESTAMP_RE = re.compile(r'^#(\d+)', re.MULTILINE)
_VALUE_RE = re.compile(r'^([01xXzZ])(.+)$', re.MULTILINE)

def parse_vcd(content: str):
    """Parse VCD with precompiled patterns."""
    timestamps = [int(m.group(1)) for m in _TIMESTAMP_RE.finditer(content)]
    ...
```

**Impact:** 10-20% faster parsing

### Array Copy Optimization

**IMPORTANT**: Most `.copy()` calls are NECESSARY for correctness.

#### ✅ NECESSARY: Protect Mutable State

```python
# NECESSARY: Prevent caller mutations
def get_records(self):
    return self._records.copy()  # Protects internal state
```

#### ✅ NECESSARY: Loop Snapshots

```python
# NECESSARY: Snapshot mutable variable in loop
for event in events:
    current_times.append(event)
    sequences[key].append(current_times.copy())  # MUST copy!
```

#### ⚠️ CAREFUL: Potential Optimization

```python
# Could use view instead of copy (advanced)
filtered = data  # View, not copy
result = filtered  # Still a view
```

**Only optimize copies after:**

1. Profiling shows it's a bottleneck
2. Understanding mutation semantics
3. Adding tests to prevent data corruption

---

## Security Best Practices

From comprehensive security audit (docs/security/security-audit-2026-01-25.md):

### Authentication

**REQUIRED** for REST API deployments:

```python
# ✅ GOOD: Production configuration
server = RESTAPIServer(
    api_key=os.environ["API_KEY"],  # Required
    cors_origins=["https://trusted-domain.com"],  # Explicit
    host="127.0.0.1",  # Localhost only
)

# ❌ BAD: Default configuration (insecure)
server = RESTAPIServer()  # No auth, CORS wildcard
```

### Cryptography

```python
# ✅ GOOD: MD5 for non-security purposes
import hashlib
cache_key = hashlib.md5(data, usedforsecurity=False).hexdigest()

# ✅ GOOD: HMAC for integrity
import hmac
signature = hmac.new(key, data, hashlib.sha256).digest()
if not hmac.compare_digest(signature, expected):
    raise SecurityError("Tampered data")
```

### Input Validation

```python
# ✅ GOOD: Validate editor command
ALLOWED_EDITORS = {"nano", "vim", "vi", "emacs", "code"}

def _get_safe_editor() -> str:
    editor = os.environ.get("EDITOR", "nano")
    editor_name = Path(editor).name.split()[0]

    if editor_name not in ALLOWED_EDITORS:
        logger.warning(f"Untrusted editor '{editor}', using nano")
        return "nano"

    return editor
```

### Subprocess Security

```python
# ✅ GOOD: List arguments (no shell=True)
result = subprocess.run(
    ["luac", "-p", "-"],
    input=code.encode("utf-8"),
    capture_output=True,
    timeout=5,  # Timeout protection
    check=False,
)

# ❌ BAD: Shell execution
result = subprocess.run(
    f"luac -p {filename}",  # Shell injection risk!
    shell=True,
)
```

### Deserialization

```python
# ✅ GOOD: HMAC-protected pickle
import pickle
import hmac
import hashlib

def save_cache(key, value, cache_key):
    data = pickle.dumps(value)
    sig = hmac.new(cache_key, data, hashlib.sha256).digest()
    with open(cache_file, "wb") as f:
        f.write(sig)
        f.write(data)

def load_cache(key, cache_key):
    with open(cache_file, "rb") as f:
        sig = f.read(32)
        data = f.read()

    expected = hmac.new(cache_key, data, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        raise SecurityError("Cache integrity check failed")

    return pickle.loads(data)
```

---

## Documentation Requirements

### Docstring Format

Use Google-style docstrings (as configured in pyproject.toml):

```python
def example_function(param1: str, param2: int = 0) -> bool:
    """Brief one-line description.

    Extended description if needed. Can span multiple lines
    and provide additional context.

    Args:
        param1: Description of first parameter.
        param2: Description of second parameter. Defaults to 0.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is empty.
        TypeError: When param2 is not an integer.

    Examples:
        >>> example_function("value", 10)
        True

    Note:
        Additional implementation notes if needed.

    References:
        IEEE 181-2011 Section X.X (if applicable)
    """
```

### Documentation Checklist

Before submitting a PR that includes new code, ensure:

- [ ] All new public functions have docstrings
- [ ] Docstrings follow Google style format
- [ ] Args section lists all parameters with descriptions
- [ ] Returns section describes the return value
- [ ] Raises section documents all exceptions (if applicable)
- [ ] Examples are included for complex functionality
- [ ] Demo README updated if behavior changes
- [ ] CHANGELOG.md updated for user-visible changes
- [ ] **IEEE references included** for measurement functions
- [ ] **Local verification passes** (`./scripts/pre-push.sh`)

---

## Git Workflow

### Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, missing semi-colons, etc.)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests
- `chore`: Maintenance tasks
- `perf`: Performance improvement

**Examples:**

```text
feat(protocols): add FlexRay decoder support
fix(loaders): correct Tektronix WFM channel parsing
docs(api): add spectral analysis examples
test(analyzers): increase rise time test coverage
perf(loaders): implement lazy imports for 145x speedup
```

### CHANGELOG Updates

**MANDATORY**: Every user-facing change MUST update CHANGELOG.md:

```markdown
## [Unreleased]

### Added
- **Feature Name** (path/to/file.py): Description with test count (42 tests)

### Changed
- **Behavior Change** (path/to/file.py): What changed and why

### Fixed
- **Bug Fix** (path/to/file.py): What was broken and how it's fixed

### Removed
- **Removed Feature** (path/to/file.py): Why it was removed

### Deprecated
- **Deprecated Feature** (path/to/file.py): Migration path

### Performance
- **Optimization** (path/to/file.py): Speedup achieved (10-30x faster)

### Security
- **Security Fix** (path/to/file.py): Vulnerability fixed
```

**Format:**

- Bold component name
- File path in parentheses
- Clear description with impact
- Test count if applicable
- Performance numbers if applicable

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes using conventional commits
   - Pre-commit hooks run automatically (lint, format, type check)
5. **Run local CI verification** (`./scripts/pre-push.sh`)
   - This is AUTOMATIC if you installed hooks via `./scripts/setup/install-hooks.sh`
   - Verifies your code against the full CI pipeline locally
6. Push to your branch (`git push origin feature/amazing-feature`)
   - Pre-push hook runs automatically (if installed)
7. Open a Pull Request

**CRITICAL:** Steps 4-5 must pass BEFORE step 6. If you skip verification, your PR will fail CI.

---

## IEEE Compliance Guidelines

When implementing measurement functions, follow IEEE standards:

- **IEEE 181-2011**: Pulse measurements (rise/fall time, slew rate)
- **IEEE 1241-2010**: ADC testing (SNR, SINAD, ENOB)
- **IEEE 2414-2020**: Jitter measurements (TIE, period jitter)

Include references to specific standard sections in docstrings:

```python
def rise_time(trace: TraceData, low: float = 0.1, high: float = 0.9) -> float:
    """Calculate rise time per IEEE 181-2011 Section 5.2.

    The rise time is the interval between the reference level instants
    when the signal crosses the specified low and high percentage levels.

    Args:
        trace: Input waveform trace.
        low: Low reference level (0-1). Default 10%.
        high: High reference level (0-1). Default 90%.

    Returns:
        Rise time in seconds.

    References:
        IEEE 181-2011 Section 5.2 "Rise Time and Fall Time"
    """
```

---

## Troubleshooting CI Failures

If CI fails after push, here's how to debug:

### 1. Run Local Verification First

```bash
# This should catch most issues
./scripts/pre-push.sh

# If tests fail locally, get detailed output
uv run pytest <failing_test> -v --tb=long
```

### 2. Common CI Failure Causes

| Failure | Local Check | Fix |
|---|---|---|
| Ruff lint | `./scripts/quality/lint.sh` | `./scripts/fix.sh` |
| Ruff format | `./scripts/quality/format.sh` | `./scripts/fix.sh` |
| MyPy | `uv run mypy src/` | Fix type errors |
| Pre-commit | `pre-commit run --all-files` | Follow error messages |
| MkDocs | `uv run mkdocs build --strict` | Fix broken links/warnings |
| Docstrings | `uv run interrogate src/oscura -f 95` | Add missing docstrings |

### 3. Environment Differences

CI runs on Ubuntu with Python 3.12 and 3.13. If tests pass locally but fail in CI:

```bash
# Check Python version
python --version

# Run tests with strict markers (like CI)
uv run pytest tests/unit -v --strict-markers --strict-config
```

---

## Versioning and Compatibility

Oscura follows [Semantic Versioning](https://semver.org/) (SemVer):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

**Stability Commitment:**

- Backwards compatibility - Maintained within major versions
- Deprecation warnings - Added before removing features
- Migration guides - Provided for major version upgrades
- Semantic versioning - Strictly followed for all releases

---

## Project Structure

```text
oscura/
├── demos/                  # Primary documentation (working examples)
│   ├── 01_waveform_analysis/       # Waveform loading and analysis
│   ├── 02_file_format_io/          # CSV, HDF5, NumPy formats
│   ├── 03_custom_daq/              # Custom DAQ loaders
│   ├── 04_serial_protocols/        # UART, SPI, I2C basics
│   ├── 05_protocol_decoding/       # Multi-protocol decoding
│   ├── 06_udp_packet_analysis/     # Network packet analysis
│   ├── 07_protocol_inference/      # Unknown protocol inference
│   ├── 08_automotive_protocols/    # CAN, OBD-II, J1939
│   ├── 09_automotive/              # Advanced automotive
│   ├── 10_timing_measurements/     # Timing and jitter
│   ├── 11_mixed_signal/            # Mixed signal validation
│   ├── 12_spectral_compliance/     # Spectral analysis (IEEE 1241)
│   ├── 13_jitter_analysis/         # Jitter (IEEE 2414)
│   ├── 14_power_analysis/          # Power quality (IEEE 1459)
│   ├── 15_signal_integrity/        # Signal integrity metrics
│   ├── 16_emc_compliance/          # EMC compliance testing
│   ├── 17_signal_reverse_engineering/  # Signal RE workflows
│   ├── 18_advanced_inference/      # Advanced inference
│   └── 19_complete_workflows/      # End-to-end workflows
├── docs/                   # API reference & technical docs
│   ├── api/                    # Generated API reference
│   └── testing/                # Testing documentation
├── src/oscura/           # Source code
│   ├── core/               # Data types, exceptions, configuration
│   ├── loaders/            # File format loaders
│   ├── analyzers/          # Signal analysis modules
│   ├── protocols/          # Protocol decoders
│   ├── inference/          # Protocol inference
│   ├── export/             # Data export formats
│   └── visualization/      # Plotting utilities
├── scripts/                # Development utilities
│   ├── setup/              # Setup and installation
│   │   ├── install-hooks.sh    # Install git hooks (REQUIRED)
│   │   └── verify.sh           # Quick verification
│   ├── quality/            # Quality checks
│   │   ├── lint.sh             # Linting only
│   │   └── format.sh           # Formatting only
│   ├── testing/            # Test utilities
│   │   └── run_coverage.sh     # Coverage report
│   ├── pre-push.sh         # Full CI verification (use before push)
│   ├── check.sh            # Lint + typecheck (use frequently)
│   ├── fix.sh              # Auto-fix all issues
│   └── test.sh             # Optimal test execution (SSOT)
└── tests/                  # Test suite
    ├── unit/               # Unit tests
    ├── integration/        # Integration tests
    └── conftest.py         # Test fixtures and configuration
```

---

## Getting Help

- **GitHub Discussions**: For questions and discussions
- **GitHub Issues**: For bugs and feature requests

---

## Recognition

Contributors are recognized in:

- The CHANGELOG.md for significant contributions
- The GitHub contributors page
- Special thanks in release notes for major features

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Code of Conduct

This project and everyone participating in it is governed by the [Oscura Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

---

Thank you for contributing to Oscura!
