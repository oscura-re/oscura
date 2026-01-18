# Integration Test Charter

**Purpose**: Define the scope and boundaries of integration tests to prevent redundancy with demos and unit tests.

**Last Updated**: 2026-01-15
**Status**: Active

---

## What Belongs in Integration Tests

An integration test belongs in this suite if it meets **ALL** of these criteria:

### ✅ Must Meet

1. **Tests 2+ modules crossing data boundaries**
   - Example: WFM loader → FFT analyzer data flow
   - Example: Signal analyzer → protocol decoder pipeline
   - Counter-example: FFT algorithm correctness (unit test)

2. **Tests scenarios NOT covered by demos**
   - Example: Malformed file handling
   - Example: Memory limits with very large files
   - Counter-example: Basic WFM loading (Demo 01 covers this)

3. **Tests error handling chains across modules**
   - Example: Loader error → Analyzer graceful degradation
   - Example: Protocol decoder framing error recovery
   - Counter-example: Single function error handling (unit test)

4. **Tests real file format edge cases**
   - Example: Tektronix vendor-specific quirks
   - Example: Truncated files, corrupted headers
   - Counter-example: Synthetic signal validation (unit test)

### ❌ Does NOT Belong

Integration tests should NOT:

1. **Duplicate demo functionality**
   - If a demo validates the workflow, write a test only for edge cases not in the demo
   - Example: Demo 01 validates WFM→FFT→measurements, don't retest this

2. **Test single-module algorithms**
   - Algorithm correctness belongs in unit tests
   - Example: FFT peak detection logic (unit test)
   - Example: UART baud rate calculation (unit test)

3. **Test vendor library functionality**
   - Don't test NumPy FFT, pandas DataFrames, scipy filters
   - Trust that vendor libraries work correctly
   - Example: NumPy FFT magnitude computation (vendor's job)

4. **Test comprehensive workflows already in demos**
   - Demos ARE integration tests
   - Only add integration tests for scenarios demos don't cover
   - Example: Demo 05 tests all protocol decoding, don't retest

---

## Examples

### GOOD Integration Tests

```python
def test_malformed_wfm_error_handling():
    """Test loader → analyzer data flow with corrupted WFM file.

    This is a GOOD integration test because:
    - Tests 2 modules (loader + analyzer)
    - Tests edge case NOT in demos (corrupted files)
    - Validates error handling chain
    """
    corrupted_wfm = create_corrupted_wfm()

    # Loader should detect corruption
    with pytest.raises(WFMFormatError):
        load(corrupted_wfm)


def test_very_large_file_memory_management():
    """Test streaming loader with 10 GB file.

    This is a GOOD integration test because:
    - Tests real-world edge case (memory limits)
    - Not practical to include in demo
    - Validates chunked loading behavior
    """
    huge_file = generate_10gb_test_file()

    # Should stream without loading entire file
    for chunk in load_streaming(huge_file):
        assert sys.getsizeof(chunk) < 100_000_000  # <100 MB chunks
```

### BAD Integration Tests (Move or Delete)

```python
def test_fft_magnitude_computation():
    """Test FFT magnitude calculation.

    This is a BAD integration test because:
    - Tests single module (analyzer only)
    - Tests NumPy functionality (vendor library)
    - Belongs in unit tests OR nowhere (trust NumPy)
    """
    signal = generate_sine(1e3)
    fft_result = compute_fft(signal)
    assert all(mag >= 0 for mag in fft_result.magnitude)  # NumPy guarantees this


def test_wfm_to_measurements():
    """Test loading WFM and computing measurements.

    This is a BAD integration test because:
    - Demo 01 (comprehensive_wfm_analysis.py) covers this comprehensively
    - No edge cases tested (just basic workflow)
    - Duplicates demo functionality
    """
    wfm = load("test.wfm")
    measurements = compute_measurements(wfm)
    assert measurements["rms"] > 0  # Already tested in Demo 01
```

---

## Quality Gate Checklist

Before adding a new integration test, verify:

- [ ] Does a demo already test this workflow?
  - If YES: Only test edge cases NOT in demo
  - If NO: Consider adding to demo first, then test edge cases

- [ ] Does this test 2+ modules?
  - If NO: Move to unit tests

- [ ] Does this test vendor library functionality?
  - If YES: Delete test (trust vendor)

- [ ] Is this an edge case not practical for demos?
  - Malformed files? YES ✅
  - Memory limits? YES ✅
  - Basic workflow? NO ❌

- [ ] Does this test error handling chains?
  - Loader error → Analyzer handling? YES ✅
  - Single function error? NO ❌ (unit test)

---

## Decision Flowchart

```
New test needed?
  │
  ├─ Tests single module? → Unit test
  │
  ├─ Tests vendor library? → Delete (trust vendor)
  │
  ├─ Workflow in demo? → Only test edge cases
  │
  ├─ Tests 2+ modules? → Integration test
  │     │
  │     ├─ Edge case? → Integration test ✅
  │     └─ Basic workflow? → Add to demo instead
  │
  └─ Error handling chain? → Integration test ✅
```

---

## Integration Test Categories

### 1. Data Flow Tests

Test data passing correctly between modules.

**Example**: `test_loader_to_analyzer_data_flow()`

- Focus: Data structures, API contracts
- Not: Algorithm correctness

### 2. Error Handling Tests

Test error propagation across module boundaries.

**Example**: `test_loader_error_analyzer_graceful_degradation()`

- Focus: Error handling chains
- Not: Single function error handling

### 3. Edge Case Tests

Test scenarios not practical for demos.

**Example**: `test_truncated_file_handling()`

- Focus: Real-world edge cases
- Not: Synthetic data validation

### 4. Real File Tests

Test real vendor file formats with quirks.

**Example**: `test_tektronix_wfm_vendor_quirks()`

- Focus: Format-specific edge cases
- Not: Basic file loading (demos cover this)

---

## Relationship to Demos

### Demos Are Integration Tests

Demos validate comprehensive workflows and serve as living integration tests.

**When to write demo vs integration test**:

|Scenario|Write Demo|Write Integration Test|
|---|---|---|
|Basic workflow|✅|❌|
|Common use case|✅|❌|
|Edge case (malformed files)|❌|✅|
|Edge case (memory limits)|❌|✅|
|Error handling|❌|✅|
|Vendor file quirks|❌|✅|

### Demo Coverage

See `DEMO_COVERAGE.md` for complete mapping of which demos cover which integration test scenarios.

---

## Enforcement

### Pre-Commit Hook

A pre-commit hook validates new integration tests against this charter:

```bash
# Run automatically on git commit
./scripts/check_integration_tests.py
```

### Code Review

All new integration tests must pass charter review:

1. Reviewer checks if workflow is in demo
2. Reviewer verifies 2+ modules tested
3. Reviewer confirms edge case not in demo
4. Reviewer validates no vendor library testing

---

## Migration Notes

**Tests removed during optimization (2026-01-15)**:

- `test_chunked_consistency.py` (434 LOC) - Tested NumPy FFT, not Oscura
- `test_wfm_to_analysis.py` redundant tests (236 LOC) - Covered by Demo 01
- Basic workflow tests merged - Demos now validate these

**Tests kept**:

- Edge case handling (malformed files, memory limits)
- Multi-module data flow validation
- Real vendor file quirks
- Error handling chains

**Total reduction**: 2,221 LOC (47% of integration tests)

---

## References

- **Test Suite Optimization Plan**: `TEST_SUITE_OPTIMIZATION_PLAN.md`
- **Demo Coverage**: `DEMO_COVERAGE.md`
- **Testing Strategy**: `tests/README.md`
- **Architecture Analysis**: `ARCHITECTURE_ANALYSIS_SUMMARY.md`

---

**Maintain this charter**: Update when adding new test categories or demo coverage expands.
