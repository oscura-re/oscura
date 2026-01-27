# Test Skip Documentation

This document explains why certain tests are skipped and under what conditions they execute.

## Overview

Total skips as of 2026-01-25: **559**

- Valid conditional skips: **133** (100% documented with inline comments)
- Investigation needed: 426 (see audit report)
- All valid skips have inline documentation following "# SKIP: Valid" pattern
- Documentation status: COMPLETE - all conditional skips documented

---

## Valid Skip Categories

### 1. Conditional Dependency Skips (97 tests)

**Pattern**: Skip only when optional dependency is NOT installed

```python
try:
    import optional_dependency
except ImportError:
    # SKIP: Valid - Optional dependency_name dependency
    # Only skip if dependency_name not installed (pip install oscura[extra])
    pytest.skip("optional_dependency not available")
```

**Rationale**: These tests verify functionality that depends on optional libraries. They run when the dependency is installed and skip gracefully when it's not available.

**Documentation Requirement**: All conditional skips MUST have inline "# SKIP: Valid" comments explaining why the skip is legitimate.

#### Optional Dependencies

| Dependency | Version | Test Coverage |
|------------|---------|---------------|
| matplotlib | 3.10.8 | Visualization tests |
| numba | 0.60.0 | Performance-optimized analyzers |
| pywavelets | 1.9.0 | Wavelet analysis |
| pyyaml | 6.0.3 | Configuration loading |
| pandas | 2.3.3 | Data aggregation |
| h5py | 3.15.1 | HDF5 file format |
| scipy | 1.17.0 | Scientific computing |
| nptdms | 1.10.0 | TDMS file format |
| networkx | 3.6.1 | Graph algorithms |

#### Example Tests

**Visualization Tests** (`tests/unit/visualization/`)

- Skip if matplotlib not available
- Run when matplotlib installed
- Test plot generation, styling, exports

**Loader Tests** (`tests/unit/loaders/`)

- Skip if format library not available (h5py, nptdms, etc.)
- Run when library installed
- Test file loading, validation, error handling

**Analysis Tests** (`tests/unit/analyzers/`)

- Skip if optimization library not available (numba, scipy)
- Run when library installed
- Test accelerated algorithms, spectral analysis

#### Why This Is Correct

1. **Graceful Degradation**: Users without optional deps can still run core tests
2. **CI Flexibility**: Different CI jobs can test different dependency combinations
3. **Development Workflow**: Developers can work on core features without installing all deps
4. **No False Failures**: Tests don't fail due to missing optional libraries

**Action Required**: NONE - Working as designed

---

### 2. Platform-Specific Skips (6 tests)

All platform-specific skips are documented with clear explanations of the platform limitation.

#### Filesystem Limitations

**Test**: Symlink creation
**Skip condition**: Platform doesn't support symlinks (e.g., Windows FAT32)
**Files**:

- Integration tests creating symlinks for test data
- Loader tests following symlinked paths

**Example**:

```python
if not os.supports_follow_symlinks:
    pytest.skip("Symlinks not supported on this system")
```

**Test**: Long filename support
**Skip condition**: Filesystem has path length limits (e.g., Windows MAX_PATH)
**Files**:

- Tests with deeply nested directories
- Tests with very long filenames

**Example**:

```python
if platform.system() == "Windows" and len(path) > 260:
    pytest.skip("Path too long for filesystem")
```

#### Why This Is Correct

1. **Platform Reality**: Some features genuinely aren't available on all platforms
2. **CI Coverage**: Linux CI has full coverage, other platforms skip unsupported tests
3. **User Experience**: Tests don't fail on Windows due to Linux-specific features

**Action Required**: NONE - Document platform requirements

---

### 3. Test Data Dependency Skips (30 tests)

**Pattern**: Skip only when required test data files are not available

```python
if not test_files:
    # SKIP: Valid - Test data dependency
    # Only skip if required WFM test files not available
    pytest.skip("No WFM files available")
```

**Common test data dependencies:**

- Tektronix WFM files (real oscilloscope captures)
- PCAP network captures (HTTP, Modbus, DNS)
- Synthetic test signals (square waves, UART, etc.)
- Ground truth data files (for validation tests)

**Why This Is Correct**:

1. **Real hardware files**: Not all developers have real oscilloscope captures
2. **Optional test data**: Core tests work with synthetic data only
3. **Large files**: Some test data files are too large for git repository
4. **Graceful degradation**: Tests run with available data, skip when unavailable

---

## Complete Skip Inventory (133 Valid Conditional Skips)

### By Category

| Category | Count | Documentation Status | Files Affected |
|----------|-------|---------------------|----------------|
| PyWavelets | 24 | ✓ 100% documented | test_wavelets.py, test_spectral.py |
| h5py | 3 | ✓ 100% documented | test_error_handling.py |
| matplotlib | 49 | ✓ 100% documented | test_plot_types.py, test_visualization_*.py |
| scipy | 2 | ✓ 100% documented | test_complete_workflows.py |
| PyYAML | 14 | ✓ 100% documented | test_config_validation.py, test_template_definition.py |
| sklearn | 3 | ✓ 100% documented | test_clustering_hypothesis.py |
| nptdms | 1 | ✓ 100% documented | test_error_handling.py |
| scapy | 0 | N/A | (future use) |
| Test data (WFM) | 15 | ✓ 100% documented | test_wfm_loading.py, test_tektronix*.py |
| Test data (PCAP) | 8 | ✓ 100% documented | test_pcap_*.py |
| Test data (general) | 5 | ✓ 100% documented | test_synthetic_*.py, test_protocol_messages.py |
| Module availability | 6 | ✓ 100% documented | Various integration tests |
| Platform-specific | 6 | ✓ 100% documented | test_edge_cases.py |
| Build tools (luac) | 2 | ✓ 100% documented | test_wireshark.py |
| **TOTAL** | **133** | **✓ 100% documented** | **68 files** |

### Documentation Template by Category

#### Optional Dependencies

**PyWavelets** (wavelets analysis):

```python
try:
    import pywt  # noqa: F401
except ImportError:
    # SKIP: Valid - Optional pywavelets dependency
    # Only skip if pywavelets not installed (pip install oscura[wavelets])
    pytest.skip("PyWavelets not installed")
```

**h5py** (HDF5 file format):

```python
try:
    import h5py
except ImportError:
    # SKIP: Valid - Optional h5py dependency
    # Only skip if h5py not installed (pip install oscura[hdf5])
    pytest.skip("h5py not installed")
```

**matplotlib** (visualization):

```python
try:
    import matplotlib  # noqa: F401
except ImportError:
    # SKIP: Valid - Optional matplotlib dependency
    # Only skip if matplotlib not installed (pip install oscura[viz])
    pytest.skip("matplotlib not available")
```

**scipy** (scientific computing):

```python
try:
    import scipy  # noqa: F401
except ImportError:
    # SKIP: Valid - Optional scipy dependency
    # Only skip if scipy not installed (core numerical library)
    pytest.skip("scipy not available")
```

**PyYAML** (configuration files):

```python
try:
    import yaml
except ImportError:
    # SKIP: Valid - Optional PyYAML dependency
    # Only skip if PyYAML not installed (configuration file support)
    pytest.skip("PyYAML not available")
```

**scikit-learn** (ML/clustering):

```python
try:
    from oscura.analyzers.patterns.clustering import cluster_patterns
except ImportError:
    # SKIP: Valid - Optional scikit-learn dependency
    # Only skip if sklearn not installed (pip install oscura[ml])
    pytest.skip("clustering module not available")
```

**nptdms** (TDMS file format):

```python
try:
    import nptdms  # noqa: F401
except ImportError:
    # SKIP: Valid - Optional nptdms dependency
    # Only skip if nptdms not installed (TDMS file format support)
    pytest.skip("nptdms not available")
```

#### Platform-Specific

**Symlinks** (filesystem feature):

```python
try:
    # Create test symlink
    symlink_path.symlink_to(target_path)
except (OSError, NotImplementedError):
    # SKIP: Valid - Platform-specific test
    # Only skip on platforms without symlink support (e.g., Windows FAT32)
    pytest.skip("Symlinks not supported on this system")
```

**Filesystem limits** (path length):

```python
if len(str(long_path)) > 260:
    # SKIP: Valid - Platform-specific filesystem feature
    # Only skip when filesystem doesn't support long paths (Windows MAX_PATH)
    pytest.skip("Path too long for filesystem")
```

#### Test Data

**WFM files** (oscilloscope captures):

```python
wfm_files = list(test_data_dir.glob("*.wfm"))
if not wfm_files:
    # SKIP: Valid - Test data dependency
    # Only skip if Tektronix WFM test files not available
    pytest.skip("No WFM files available")
```

**PCAP files** (network captures):

```python
pcap_file = test_data_dir / "http_traffic.pcap"
if not pcap_file.exists():
    # SKIP: Valid - Test data dependency
    # Only skip if PCAP test files not available
    pytest.skip("HTTP PCAP not available")
```

**Ground truth data** (validation):

```python
if "ground_truth" not in test_data:
    # SKIP: Valid - Test data dependency
    # Only skip if ground truth validation data not available
    pytest.skip("Ground truth not available")
```

#### Build Tools

**luac** (Lua compiler):

```python
import shutil

if not shutil.which("luac"):
    # SKIP: Valid - Build tool dependency
    # Only skip if luac (Lua compiler) not installed in PATH
    pytest.skip("luac not available")
```

---

## How to Add a Skip

### Good Skip (with rationale)

```python
def test_advanced_feature():
    """Test feature that requires optional dependency."""
    try:
        import optional_lib
    except ImportError:
        pytest.skip("optional_lib required for advanced_feature")

    # Test code here
    result = advanced_feature_using_optional_lib()
    assert result is not None
```

### Bad Skip (no rationale, always skips)

```python
@pytest.mark.skip("TODO: implement this")  # ❌ BAD
def test_future_feature():
    pass
```

**Instead, use**:

```python
@pytest.mark.xfail(reason="Feature not yet implemented", strict=False)
def test_future_feature():
    pass
```

---

## Skip Guidelines

### DO:

✅ Skip when optional dependency is not available
✅ Skip when platform lacks required feature (symlinks, etc.)
✅ Skip when test data is genuinely unavailable (document why)
✅ Provide clear, actionable skip reasons
✅ Use conditional skips (try/except ImportError)

### DON'T:

❌ Skip tests for installed dependencies (run audit to find these)
❌ Skip tests with reason "TODO" (use @pytest.mark.xfail instead)
❌ Skip tests permanently without explanation
❌ Skip tests that could be fixed by generating test data
❌ Skip tests because they're flaky (fix the test)

---

## Skip Reason Best Practices

### Good Skip Reasons

```python
pytest.skip("h5py required for HDF5 file loading")  # Clear dependency
pytest.skip("Symlinks not supported on this filesystem")  # Clear limitation
pytest.skip("Test requires >4GB RAM")  # Clear resource requirement
```

### Bad Skip Reasons

```python
pytest.skip("broken")  # What's broken? Why?
pytest.skip("TODO")  # Use xfail instead
pytest.skip("doesn't work")  # Why? On what platform?
```

### Skip Reason Template

```
pytest.skip("<DEPENDENCY/FEATURE> required for <FUNCTIONALITY>")
```

Examples:

- `pytest.skip("matplotlib required for plot generation")`
- `pytest.skip("Symlinks required for path testing")`
- `pytest.skip("GPU required for CUDA acceleration tests")`

---

## Reviewing Skipped Tests

### When to Question a Skip

1. **Dependency is installed**: Run audit to identify

   ```bash
   python3 .claude/audit_skipped_tests.py
   ```

2. **Skip reason is vague**: Update with specific reason

3. **Test has been skipped for >6 months**: Review if still relevant

4. **Skip has no try/except**: Should be @pytest.mark.skipif

### Periodic Audit Schedule

- **Weekly**: Check for new skips in PRs
- **Monthly**: Run full skip audit
- **Quarterly**: Review all "investigate" skips for resolution

---

## CI/CD Integration

### Pre-commit Hook

Validates new skips have proper reasons:

```python
# .claude/hooks/validate_skips.py
def check_skip_reason(skip_line):
    if 'pytest.skip("")' in skip_line:
        raise ValueError("Skip reason cannot be empty")
    if 'pytest.skip("TODO")' in skip_line:
        raise ValueError("Use @pytest.mark.xfail instead of TODO skip")
```

### CI Skip Report

GitHub Actions generates skip report:

```yaml
- name: Generate Skip Report
  run: |
    python3 .claude/audit_skipped_tests.py
    cat .claude/SKIPPED_TESTS_AUDIT_2026-01-25.md >> $GITHUB_STEP_SUMMARY
```

---

## FAQ

### Q: Why not just install all optional dependencies in CI?

**A**: Different users have different environments. Some dependencies are large (matplotlib, scipy), platform-specific (Windows-only libs), or have complex installation requirements. Conditional skips allow testing both with and without optional deps.

### Q: Why not use @pytest.mark.skipif instead of try/except?

**A**: Both patterns are valid:

- `@pytest.mark.skipif`: Good for module-level checks
- `try/except`: Good for inline checks, more flexible

We use both depending on context.

### Q: How do I run skipped tests?

**A**: Install the required dependency:

```bash
uv pip install matplotlib  # For visualization tests
uv pip install h5py        # For HDF5 loader tests
./scripts/test.sh          # Skipped tests now run
```

### Q: Should I remove skips for dependencies in pyproject.toml?

**A**: YES! If the dependency is in `[project.dependencies]` or required `[project.optional-dependencies]`, tests should NOT skip. Run the audit to find these.

### Q: What's the difference between skip and xfail?

**A**:

- **skip**: Test cannot run (missing dependency, platform limitation)
- **xfail**: Test can run but is expected to fail (known bug, unimplemented feature)

---

## Maintenance

This document is maintained by the test infrastructure team. For questions or issues:

1. Check skip audit: `python3 .claude/audit_skipped_tests.py`
2. Review this documentation
3. Open issue if skip seems incorrect
4. Submit PR to fix or document skip

**Last Updated**: 2026-01-25
**Next Audit Due**: 2026-02-25
