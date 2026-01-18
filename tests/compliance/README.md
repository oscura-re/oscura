# Compliance Tests

This directory contains tests that verify Oscura's compliance with industry standards and specifications.

## Standards Tested

Oscura implements measurements according to these standards:

- **IEEE 181-2011** - Pulse measurements (rise/fall time, overshoot, slew rate)
- **IEEE 1057-2017** - Digitizer characterization and timing analysis
- **IEEE 1241-2010** - ADC testing (SNR, SINAD, ENOB)
- **IEEE 1459** - Power measurements and power quality
- **IEEE 2414-2020** - Jitter measurements (TIE, period jitter)
- **IEC 61000-4-7** - Harmonics analysis and power quality
- **JEDEC** - Digital timing (setup/hold time)

## Purpose

Compliance tests verify that Oscura measurements:

1. Use correct algorithms as specified in standards
2. Produce results within acceptable tolerance
3. Handle edge cases as defined by standards
4. Report measurements in standard units

## Organization

```
compliance/
├── test_ieee_181.py         # Pulse measurements
├── test_ieee_1057.py        # Digitizer characterization
├── test_ieee_1241.py        # ADC testing
├── test_ieee_1459.py        # Power measurements
├── test_ieee_2414.py        # Jitter measurements
├── test_iec_61000.py        # Harmonics
└── test_jedec_timing.py     # Digital timing
```

## Running Compliance Tests

```bash
# Run all compliance tests
uv run pytest tests/compliance/

# Run specific standard tests
uv run pytest tests/compliance/test_ieee_181.py

# Generate compliance report
uv run pytest tests/compliance/ --html=compliance_report.html
```

## Test Structure

### Reference Data

Compliance tests use:

- Known reference signals with calculated expected values
- Standard test vectors from IEEE specifications
- Synthetic signals with precise ground truth

### Tolerance Checks

```python
@pytest.mark.compliance
def test_rise_time_ieee_181():
    """Verify rise time measurement per IEEE 181-2011."""
    # Generate reference signal with known rise time
    signal, ground_truth = generate_pulse_with_rise_time(
        rise_time=10e-9,  # 10 ns
        sample_rate=10e9   # 10 GHz
    )

    # Measure using Oscura
    measured = osc.rise_time(signal, method="ieee_181")

    # Verify within tolerance (IEEE 181 allows 2%)
    relative_error = abs(measured - ground_truth) / ground_truth
    assert relative_error < 0.02, f"Rise time error {relative_error*100:.2f}%"
```

## Test Markers

- `@pytest.mark.compliance` - Compliance test
- `@pytest.mark.ieee_181` - IEEE 181 standard
- `@pytest.mark.ieee_1241` - IEEE 1241 standard
- `@pytest.mark.slow` - Long-running test

## Guidelines

1. **Reference Values**: Use documented test vectors or calculated ground truth
2. **Tolerances**: Apply standard-specified tolerances
3. **Units**: Verify measurements use correct SI units
4. **Edge Cases**: Test boundary conditions specified in standards
5. **Documentation**: Include standard section references in docstrings

## Validation

Compliance is validated through:

- Known test vectors from standards documents
- Cross-validation with reference implementations
- Synthetic signals with analytical ground truth
- Comparison with calibrated measurement equipment

## See Also

- [Testing Guide](../../docs/testing/test-suite-guide.md) - Complete testing documentation
- [Test Suite Overview](../README.md) - Test suite organization
