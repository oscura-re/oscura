# Comparison and Validation Testing

**Compare signals against references, limits, and masks for quality assurance.**

This section contains 4 demonstrations showing how to perform golden reference comparison, limit testing, mask testing, and regression testing. Essential for production testing, quality assurance, and automated validation workflows.

---

## Prerequisites

Before running these demonstrations, ensure you have:

- **Completed Basic Analysis** - Run `demonstrations/02_basic_analysis/` first
- **Understanding of Statistics** - Mean, standard deviation, correlation
- **Signal Generation Knowledge** - Familiarity with `demonstrations/17_signal_generation/`
- **Quality Concepts** - Pass/fail criteria, tolerance bands, specifications

Check your setup:

```bash
python demonstrations/02_basic_analysis/01_waveform_measurements.py
# Should show successful measurements
```

---

## Demonstrations

| Demo                  | Time       | Difficulty       | Focus                               |
| --------------------- | ---------- | ---------------- | ----------------------------------- |
| 01_golden_reference   | 15 min     | Intermediate     | Compare against reference waveforms |
| 02_limit_testing      | 10 min     | Intermediate     | Pass/fail against parameter limits  |
| 03_mask_testing       | 15 min     | Advanced         | Waveform mask conformance testing   |
| 04_regression_testing | 10 min     | Intermediate     | Detect changes from baseline        |
| **Total**             | **50 min** | **Intermediate** | **Comparison testing**              |

---

## Learning Path

Complete these demonstrations in order for comprehensive testing knowledge:

```
01_golden_reference.py    02_limit_testing.py    03_mask_testing.py    04_regression_testing.py
         ↓                        ↓                       ↓                        ↓
Reference comparison      Limit testing          Mask testing         Regression detection
Correlation/RMSE/MAE      Min/max criteria       Eye diagrams         Baseline comparison
```

### Estimated Time: 50 minutes

---

## Key Concepts

This section teaches:

1. **Golden Reference Comparison** - Statistical comparison against reference waveforms
2. **Limit Testing** - Pass/fail testing against parameter specifications
3. **Mask Testing** - Waveform conformance to amplitude/timing masks
4. **Regression Testing** - Detect signal changes from baseline
5. **Statistical Metrics** - Correlation, RMSE, MAE, peak error

---

## Running Demonstrations

### Option 1: Run Individual Demo

```bash
# From the project root
python demonstrations/18_comparison_testing/01_golden_reference.py

# Or from the demo directory
cd demonstrations/18_comparison_testing
python 01_golden_reference.py
```

### Option 2: Run All Comparison Testing Demos

```bash
# From the project root
python demonstrations/18_comparison_testing/01_golden_reference.py && \
python demonstrations/18_comparison_testing/02_limit_testing.py && \
python demonstrations/18_comparison_testing/03_mask_testing.py && \
python demonstrations/18_comparison_testing/04_regression_testing.py
```

### Option 3: Validate All Demonstrations

```bash
# From the project root
python demonstrations/validate_all.py
```

---

## What You'll Learn

### Demo 01: Golden Reference Comparison

**File**: `01_golden_reference.py`

**Demonstrates**:

- Golden waveform comparison
- Statistical similarity metrics (correlation, RMSE, MAE)
- Pass/fail criteria and tolerance bands
- Difference visualization and analysis
- Template matching for conformance testing

**What you'll do**:

1. Compare measured signals against golden references
2. Calculate correlation coefficients for similarity
3. Measure RMSE (root mean square error)
4. Compute MAE (mean absolute error)
5. Apply pass/fail criteria based on metrics

**Capabilities**:

- Correlation coefficient calculation
- RMSE (root mean square error)
- MAE (mean absolute error)
- Peak error detection
- Pass/fail criteria evaluation

**Related Demos**:

- `18_comparison_testing/02_limit_testing.py` - Limit testing
- `18_comparison_testing/03_mask_testing.py` - Mask testing
- `02_basic_analysis/01_waveform_measurements.py` - Measurements

---

### Demo 02: Limit Testing

**File**: `02_limit_testing.py`

**Demonstrates**:

- Parameter limit testing (min/max)
- Multi-parameter validation
- Tolerance band checking
- Specification compliance testing
- Automated pass/fail reporting

**What you'll do**:

1. Define parameter limits from specifications
2. Test signals against multiple limit criteria
3. Apply tolerance bands to measurements
4. Generate pass/fail reports
5. Track limit violations over time

**Capabilities**:

- Min/max limit checking
- Tolerance band validation
- Multi-parameter testing
- Specification compliance
- Automated reporting

**Related Demos**:

- `18_comparison_testing/01_golden_reference.py` - Reference comparison
- `19_standards_compliance/01_ieee_181.py` - Standards compliance
- `12_quality_tools/01_measurement_validation.py` - Validation

---

### Demo 03: Mask Testing

**File**: `03_mask_testing.py`

**Demonstrates**:

- Amplitude/timing mask definition
- Eye diagram mask testing
- Rise/fall time masks
- Overshoot/undershoot limits
- Mask violation detection and reporting

**What you'll do**:

1. Define amplitude and timing masks
2. Test waveforms against eye diagram masks
3. Detect and report mask violations
4. Visualize waveforms with mask overlays
5. Generate mask compliance reports

**Capabilities**:

- Custom mask definition
- Eye diagram testing
- Violation detection
- Mask overlay visualization
- Compliance reporting

**Related Demos**:

- `18_comparison_testing/01_golden_reference.py` - Reference testing
- `02_basic_analysis/01_waveform_measurements.py` - Waveform measurements
- `03_protocol_decoding/01_serial_comprehensive.py` - Serial protocols

---

### Demo 04: Regression Testing

**File**: `04_regression_testing.py`

**Demonstrates**:

- Baseline signal capture
- Change detection from baseline
- Regression threshold configuration
- Automated regression detection
- Trend analysis over time

**What you'll do**:

1. Establish baseline signals
2. Compare new signals against baseline
3. Detect significant changes
4. Configure regression thresholds
5. Track signal drift over time

**Capabilities**:

- Baseline management
- Change detection
- Threshold configuration
- Automated alerts
- Trend analysis

**Related Demos**:

- `18_comparison_testing/01_golden_reference.py` - Reference comparison
- `12_quality_tools/02_statistical_validation.py` - Statistical validation
- `09_batch_processing/01_batch_analysis.py` - Batch processing

---

## Troubleshooting

### "Reference comparison fails but signals look identical"

**Solution**: Check for timing alignment and DC offset:

```python
# Remove DC offset first
test_signal = test_signal - np.mean(test_signal.data)
ref_signal = ref_signal - np.mean(ref_signal.data)

# Align signals in time
aligned_test = align_signals(test_signal, ref_signal)

# Now compare
correlation = compare_signals(aligned_test, ref_signal)
```

### "Limit test fails with borderline values"

**Solution**: Account for measurement uncertainty:

```python
# Bad: No measurement uncertainty
limit = 5.0
assert measurement == 5.0  # May fail due to precision

# Good: Include tolerance
limit = 5.0
tolerance = 0.01  # 1% tolerance
assert abs(measurement - limit) <= tolerance
```

### "Mask testing shows false violations"

**Solution**: Ensure mask coordinates match signal units:

```python
# Check voltage scale
mask_voltage = mask.voltage * signal.metadata.vertical_scale

# Check time base
mask_time = mask.time * signal.metadata.time_scale

# Apply scaled mask
violations = check_mask(signal, mask_voltage, mask_time)
```

### "Regression detection too sensitive"

**Solution**: Adjust threshold based on signal variability:

```python
# Calculate signal variance
signal_std = np.std(baseline_signals, axis=0)

# Set threshold as multiple of standard deviation
threshold = 3.0 * signal_std  # 3-sigma threshold

# Detect regressions beyond threshold
regressions = detect_changes(new_signal, baseline, threshold)
```

---

## Next Steps

### If You Want to...

| Goal                            | Next Demo                                                 | Path                 |
| ------------------------------- | --------------------------------------------------------- | -------------------- |
| Validate against IEEE standards | `19_standards_compliance/01_ieee_181.py`                  | Standards compliance |
| Generate test signals           | `17_signal_generation/01_signal_builder_comprehensive.py` | Signal generation    |
| Build production test workflows | `16_complete_workflows/02_production_testing.py`          | Complete workflows   |
| Perform batch validation        | `09_batch_processing/01_batch_analysis.py`                | Batch processing     |

### Recommended Next Sections

1. **Standards Compliance** (19_standards_compliance/)
   - IEEE 181 pulse measurements
   - IEEE 1241 ADC testing
   - IEEE 1459 power quality
   - IEEE 2414 PHY compliance

2. **Complete Workflows** (16_complete_workflows/)
   - End-to-end testing workflows
   - Production testing automation
   - Real-world case studies

3. **Batch Processing** (09_batch_processing/)
   - Automated batch validation
   - Large-scale testing
   - Statistical analysis

---

## Understanding Comparison Testing

### Statistical Similarity Metrics

Different metrics reveal different aspects of similarity:

| Metric      | Range   | Meaning                 | Best For            |
| ----------- | ------- | ----------------------- | ------------------- |
| Correlation | -1 to 1 | Linear relationship     | Shape similarity    |
| RMSE        | 0 to ∞  | Average error magnitude | Overall difference  |
| MAE         | 0 to ∞  | Average absolute error  | Robust to outliers  |
| Peak Error  | 0 to ∞  | Maximum difference      | Worst-case analysis |

**Interpretation**:

- **Correlation = 1.0** - Perfect positive correlation (identical shape)
- **Correlation = 0.0** - No linear relationship
- **Correlation = -1.0** - Perfect negative correlation (inverted)
- **RMSE/MAE = 0.0** - Identical signals
- **Low RMSE/MAE** - Similar signals
- **High RMSE/MAE** - Different signals

### Mask Testing Types

Different masks for different applications:

1. **Eye Diagram Masks** - Digital signal integrity
   - Tests bit transitions
   - Ensures adequate eye opening
   - Used in high-speed serial protocols

2. **Rise/Fall Time Masks** - Transition quality
   - Validates slew rates
   - Detects overshoot/undershoot
   - Used in clock quality testing

3. **Amplitude Masks** - Signal levels
   - Ensures proper voltage levels
   - Detects clipping or distortion
   - Used in analog signal testing

4. **Custom Masks** - Application-specific
   - User-defined boundaries
   - Complex shapes
   - Specialized testing

### Regression Testing Strategy

Effective regression testing requires:

1. **Good Baseline** - Capture baseline from known-good signals
2. **Appropriate Threshold** - Balance sensitivity vs false alarms
3. **Multiple Baselines** - Account for normal variation
4. **Trending** - Track changes over time
5. **Root Cause Analysis** - Investigate significant regressions

---

## Best Practices

### Golden Reference Testing

1. **Use Known-Good References** - Capture references from validated hardware
2. **Document Capture Conditions** - Record test conditions for reference
3. **Update References Deliberately** - Only update with approved changes
4. **Multiple References** - Capture min/typ/max references

### Limit Testing

1. **Use Datasheet Limits** - Base limits on manufacturer specifications
2. **Include Margins** - Add safety margin to limits (e.g., 80% of max)
3. **Test Multiple Parameters** - Comprehensive parameter coverage
4. **Log All Results** - Track pass/fail history

### Mask Testing

1. **Standard Masks** - Use industry-standard masks when available
2. **Margin Testing** - Test against tighter masks for margin analysis
3. **Visual Verification** - Review mask violations visually
4. **Statistical Analysis** - Track violation rate over time

### Regression Testing

1. **Stable Baseline** - Ensure baseline represents normal operation
2. **Statistical Thresholds** - Use statistical methods (e.g., 3-sigma)
3. **Regular Updates** - Update baseline with approved changes
4. **Automated Alerts** - Notify on regression detection

---

## Advanced Techniques

### Multi-Signal Comparison

Compare multiple signals simultaneously:

```python
# Compare array of test signals against reference
test_signals = [signal1, signal2, signal3, signal4]
reference = golden_reference

results = []
for test in test_signals:
    correlation = correlate(test, reference)
    rmse = calculate_rmse(test, reference)
    results.append({
        'correlation': correlation,
        'rmse': rmse,
        'pass': correlation > 0.95 and rmse < 0.1
    })

# Statistical summary
pass_rate = sum(r['pass'] for r in results) / len(results)
```

### Dynamic Mask Generation

Create masks from signal statistics:

```python
# Generate mask from signal population
signals = load_baseline_signals()

# Calculate mean and std dev at each time point
mean_trace = np.mean([s.data for s in signals], axis=0)
std_trace = np.std([s.data for s in signals], axis=0)

# Create mask as ±3σ from mean
upper_mask = mean_trace + 3 * std_trace
lower_mask = mean_trace - 3 * std_trace

# Test new signal against dynamic mask
violations = check_mask(new_signal, upper_mask, lower_mask)
```

### Adaptive Thresholds

Adjust thresholds based on signal characteristics:

```python
# Calculate adaptive threshold
signal_snr = calculate_snr(signal)

if signal_snr > 40:  # High SNR
    threshold = 0.01  # Strict threshold
elif signal_snr > 20:  # Medium SNR
    threshold = 0.05  # Moderate threshold
else:  # Low SNR
    threshold = 0.10  # Relaxed threshold

# Apply adaptive threshold
is_regression = difference > threshold
```

### Trend Analysis

Track signal changes over time:

```python
# Collect measurements over time
measurements = []
timestamps = []

for signal in signal_sequence:
    measurements.append(measure_parameter(signal))
    timestamps.append(signal.metadata.timestamp)

# Detect trends
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(
    timestamps, measurements
)

# Alert if significant trend
if abs(slope) > trend_threshold and p_value < 0.05:
    print(f"Significant trend detected: {slope:.3e} units/sec")
```

---

## Tips for Success

### Choose Appropriate Metrics

Match metric to test objective:

```python
# For shape similarity (insensitive to amplitude)
use_correlation = True

# For absolute accuracy
use_rmse = True

# For robust comparison (insensitive to outliers)
use_mae = True

# For worst-case analysis
use_peak_error = True
```

### Handle Signal Alignment

Ensure signals are properly aligned:

```python
# Cross-correlation based alignment
lag = find_alignment_lag(test_signal, reference)
aligned_test = shift_signal(test_signal, lag)

# Verify alignment
correlation_before = correlate(test_signal, reference)
correlation_after = correlate(aligned_test, reference)
assert correlation_after > correlation_before
```

### Document Test Criteria

Clearly document pass/fail criteria:

```python
test_criteria = {
    'correlation': {'min': 0.95, 'description': 'Shape similarity'},
    'rmse': {'max': 0.1, 'description': 'RMS error in volts'},
    'peak_error': {'max': 0.5, 'description': 'Maximum error in volts'},
    'test_date': datetime.now().isoformat(),
    'reference_id': 'REF_2024_001'
}

# Store with results
results['criteria'] = test_criteria
```

---

## Summary

The Comparison and Validation Testing section covers:

| Demo                  | Focus                | Outcome                        |
| --------------------- | -------------------- | ------------------------------ |
| 01_golden_reference   | Reference comparison | Statistical similarity testing |
| 02_limit_testing      | Parameter limits     | Pass/fail against specs        |
| 03_mask_testing       | Waveform masks       | Conformance testing            |
| 04_regression_testing | Change detection     | Baseline comparison            |

After completing these 50-minute demonstrations, you'll understand:

- How to compare signals against golden references using statistical metrics
- How to perform limit testing against parameter specifications
- How to use mask testing for waveform conformance
- How to detect regressions from baseline signals
- How to build automated validation workflows
- How to choose appropriate comparison metrics for different scenarios

**Ready to start?** Run this to explore golden reference comparison:

```bash
python demonstrations/18_comparison_testing/01_golden_reference.py
```

Happy testing and validation!
