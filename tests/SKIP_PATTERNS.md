# Test Skip Patterns and Documentation Standards

This document defines the patterns and documentation standards for `pytest.skip()` calls in the Oscura test suite.

## Skip Categories

### 1. Valid Conditional Skips (✓ ALLOWED)

These skips are **legitimate** and **must be documented** with inline comments.

#### Optional Dependencies

Tests that require optional dependencies (installable via extras):

```python
try:
    import h5py
except ImportError:
    # SKIP: Valid - Optional h5py dependency
    # Only skip if h5py not installed (pip install oscura[hdf5])
    pytest.skip("h5py not installed")
```

**Common optional dependencies:**

- `h5py` - HDF5 file support (`pip install oscura[hdf5]`)
- `pywavelets` - Wavelet analysis (`pip install oscura[wavelets]`)
- `scapy` - Network/PCAP analysis (`pip install oscura[network]`)
- `scikit-learn` - ML/clustering (`pip install oscura[ml]`)
- `numba` - Performance acceleration (`pip install oscura[performance]`)
- `matplotlib` - Visualization (`pip install oscura[viz]`)

#### Platform-Specific Tests

Tests that only run on specific platforms:

```python
import sys

if sys.platform != "linux":
    # SKIP: Valid - Platform-specific test
    # Only skip on non-Linux platforms (requires /dev/shm)
    pytest.skip("Linux-only test")
```

#### Test Data Dependencies

Tests that require specific test data files:

```python
if not test_files:
    # SKIP: Valid - Test data dependency
    # Only skip if sigrok test files not loaded successfully
    pytest.skip("No sigrok files loaded successfully")
```

### 2. Invalid Skips (✗ NOT ALLOWED)

These skips are **not legitimate** and should be removed:

#### Incomplete Tests (TODO)

```python
# ✗ INVALID - Remove or complete test
pytest.skip("TODO: implement later")
```

**Action**: Complete the test or remove it entirely.

#### Implementation Gaps

```python
# ✗ INVALID - Fix the implementation
pytest.skip("Feature not implemented yet")
```

**Action**: Implement the feature or file an issue.

#### Debugging/Temporary Skips

```python
# ✗ INVALID - Debug and fix
pytest.skip("Fails intermittently, needs investigation")
```

**Action**: Debug the failure, fix the root cause.

## Documentation Standards

### Required Format

All valid conditional skips **MUST** include a two-line comment:

```python
# SKIP: Valid - <category>
# <clear explanation of when skip occurs>
pytest.skip("<actionable reason>")
```

**Components:**

1. **Line 1**: `# SKIP: Valid - <category>` (marks as documented valid skip)
2. **Line 2**: `# <explanation>` (when/why skip occurs)
3. **Skip call**: `pytest.skip("<reason>")` (actionable reason string)

### Examples by Category

#### Optional Dependency

```python
# SKIP: Valid - Optional h5py dependency
# Only skip if h5py not installed (pip install oscura[hdf5])
pytest.skip("h5py not installed")
```

#### Platform-Specific

```python
# SKIP: Valid - Platform-specific test
# Only skip on Windows (POSIX-only feature)
pytest.skip("Unix-only test")
```

#### Test Data Dependency

```python
# SKIP: Valid - Test data dependency
# Only skip if test signal generator not available
pytest.skip("1MHz square wave not available")
```

#### Module Not Available

```python
# SKIP: Valid - Optional clustering module (scikit-learn)
# Only skip if sklearn not installed (pip install oscura[ml])
pytest.skip("clustering module not available")
```

### Actionable Skip Reasons

Skip reason strings should be **actionable** - tell the user what to do:

**Good:**

- `"h5py not installed (pip install oscura[hdf5])"`
- `"Requires matplotlib ≥3.0 for 3D plotting"`
- `"Linux-only test (requires /dev/shm)"`
- `"Test data missing: run scripts/test-data/generate.py"`

**Bad:**

- `"TODO"` (not actionable)
- `"Doesn't work"` (not clear)
- `"Skip this"` (no reason)
- `"Failed"` (should raise exception instead)

## Detection and Enforcement

### Automated Analysis

Use the provided scripts to audit skips:

```bash
# Find all skips and categorize them
python3 analyze_valid_skips.py

# Add documentation to valid conditional skips
python3 add_skip_documentation.py

# Verify all skips are documented
grep -r "pytest.skip" tests --include="*.py" | grep -v "# SKIP: Valid"
```

### Pre-Commit Validation

The pre-commit hook validates:

1. All conditional skips in `try/except` blocks are documented
2. No TODO/WIP skip reasons exist
3. Skip reasons are actionable (>5 characters)

## Statistics (as of 2026-01-25)

- **Total pytest.skip() calls**: 559
- **Valid conditional skips**: 133 (100% documented)
- **Optional dependency skips**: 97
- **Test data dependency skips**: 30
- **Platform-specific skips**: 6
- **Invalid skips identified**: 426 (investigation needed)

## Migration Guide

### Before (Undocumented)

```python
def test_hdf5_export(signal):
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not installed")

    # test code...
```

### After (Documented)

```python
def test_hdf5_export(signal):
    try:
        import h5py
    except ImportError:
        # SKIP: Valid - Optional h5py dependency
        # Only skip if h5py not installed (pip install oscura[hdf5])
        pytest.skip("h5py not installed")

    # test code...
```

## Common Patterns

### Optional Import Pattern

```python
def test_feature():
    try:
        from oscura.optional_module import feature
    except ImportError:
        # SKIP: Valid - Optional <module> dependency
        # Only skip if <module> not available (pip install oscura[<extra>])
        pytest.skip("<module> not available")

    # Test code using feature
    result = feature()
    assert result is not None
```

### Platform Check Pattern

```python
import sys
import pytest

@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-only test (requires fork)"
)
def test_posix_feature():
    # Test code...
    pass
```

### Test Data Check Pattern

```python
def test_with_data(test_data_dir):
    files = list(test_data_dir.glob("*.csv"))
    if not files:
        # SKIP: Valid - Test data dependency
        # Only skip if CSV test files not generated
        pytest.skip("No CSV test files available")

    # Test code using files
    for f in files:
        process(f)
```

## Best Practices

1. **Document immediately** - Add skip documentation when writing the test
2. **Be specific** - Explain the exact dependency/requirement
3. **Provide action** - Tell user how to enable the test
4. **Use extras** - Reference pip extras for optional dependencies
5. **Avoid TODO** - Complete tests or remove them
6. **Test first** - Write tests before implementing features
7. **Fail fast** - Raise exceptions for errors, skip only for missing optionals

## Validation Checklist

Before committing tests with skips:

- [ ] All skips have `# SKIP: Valid` documentation
- [ ] Skip reasons are actionable (tell user what to install/configure)
- [ ] No TODO/WIP skip reasons
- [ ] Optional dependencies referenced in skip reason match `pyproject.toml` extras
- [ ] Platform-specific skips use `@pytest.mark.skipif` decorator
- [ ] Test data skips explain how to generate missing data

## See Also

- [SKIP_DOCUMENTATION.md](SKIP_DOCUMENTATION.md) - Complete list of documented skips
- [conftest.py](conftest.py) - Shared fixtures and test configuration
- [../pyproject.toml](../pyproject.toml) - Optional dependencies extras
- [../scripts/test.sh](../scripts/test.sh) - Test execution (SSOT)
