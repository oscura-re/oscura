# Integration Tests

This directory contains integration tests that verify Oscura components work together correctly.

## Purpose

Integration tests verify:

- Multi-module workflows
- End-to-end processing pipelines
- File loading and analysis flows
- Protocol decoding with real captures
- Report generation workflows

## Organization

```
integration/
├── test_analysis_workflows.py    # Complete analysis pipelines
├── test_loader_analysis.py       # Load → Analyze → Export
├── test_protocol_workflows.py    # Protocol detection and decoding
├── test_real_captures.py         # Real oscilloscope data
└── test_reporting_pipeline.py    # Analysis → Report generation
```

## Running Integration Tests

```bash
# Run all integration tests
uv run pytest tests/integration/

# Run specific test file
uv run pytest tests/integration/test_real_captures.py

# Run with markers
uv run pytest tests/integration/ -m "not slow"

# Verbose output
uv run pytest tests/integration/ -v -s
```

## Test Data

Integration tests use:

- Real capture files from `test_data/real_captures/`
- Synthetic test data from `test_data/synthetic/`
- Multi-format test files

## Test Markers

- `@pytest.mark.integration` - Integration test
- `@pytest.mark.slow` - Long-running test (>5s)
- `@pytest.mark.requires_real_data` - Needs real capture files

## Writing Integration Tests

### Test Structure

```python
@pytest.mark.integration
def test_complete_workflow():
    """Test complete analysis workflow."""
    # Load data
    trace = osc.load("test_data/real_captures/waveforms/small/waveform_1.wfm")

    # Analyze
    freq = osc.frequency(trace)
    rise_time = osc.rise_time(trace)

    # Verify workflow
    assert freq > 0
    assert rise_time > 0
```

### Real Data Tests

```python
@pytest.mark.integration
@pytest.mark.requires_real_data
def test_tektronix_wfm_analysis(real_captures_dir):
    """Test analysis with real Tektronix capture."""
    wfm_file = real_captures_dir / "waveforms/small/waveform_1.wfm"

    # Load and analyze
    trace = osc.load(wfm_file)
    measurements = {
        "frequency": osc.frequency(trace),
        "amplitude": osc.amplitude(trace),
        "rise_time": osc.rise_time(trace)
    }

    # Verify all measurements succeeded
    assert all(m is not None for m in measurements.values())
```

## Guidelines

1. **Multi-Module**: Test interactions between 2+ modules
2. **Real Data**: Use actual capture files when possible
3. **Timeouts**: Set appropriate timeouts for long operations
4. **Cleanup**: Clean up any generated files after tests
5. **Assertions**: Verify complete workflows, not just individual steps

## See Also

- [Testing Guide](../../docs/testing/test-suite-guide.md) - Complete testing documentation
- [Test Suite Overview](../README.md) - Test suite organization
- [Unit Tests](../unit/README.md) - Unit test suite
- [Test Charter](TEST_CHARTER.md) - Integration test scope definition
- [Demo Coverage](DEMO_COVERAGE.md) - Demo-to-test mapping
