# Quality Tools and Validation

**Automated signal quality assessment, anomaly detection, and smart recommendations.**

This section contains 4 demonstrations designed to teach you how to assess signal quality, detect anomalies automatically, implement warning systems, and generate smart analysis recommendations. Perfect for production validation, automated testing, and ensuring measurement reliability.

---

## Prerequisites

See [main demonstrations README](../README.md#installation) for installation instructions.

---

## Learning Path

These demonstrations are designed to be completed **in order**. Each builds on concepts from the previous one:

```
01_ensemble_methods.py → 02_quality_scoring.py → 03_warning_system.py → 04_recommendations.py
        ↓                       ↓                       ↓                       ↓
  Multiple methods       Comprehensive scoring    Automatic detection    Smart suggestions
  Majority voting        SNR/distortion          Threshold config       Best practices
  Outlier rejection      Quality thresholds      Severity levels        Troubleshooting
```

### Estimated Time

| Demo                | Time       | Difficulty       | Topics                                   |
| ------------------- | ---------- | ---------------- | ---------------------------------------- |
| 01_ensemble_methods | 15 min     | Intermediate     | Multiple methods, voting, outliers       |
| 02_quality_scoring  | 15 min     | Intermediate     | SNR, distortion, scoring algorithms      |
| 03_warning_system   | 15 min     | Intermediate     | Anomaly detection, warnings, severity    |
| 04_recommendations  | 15 min     | Intermediate     | Smart suggestions, optimization guidance |
| **Total**           | **60 min** | **Intermediate** | **Quality assurance mastery**            |

---

## Demonstrations

### Demo 01: Ensemble Measurement Methods

**File**: `01_ensemble_methods.py`

**What it teaches**:

- Using multiple measurement methods for the same parameter
- Majority voting and weighted averaging techniques
- Outlier rejection algorithms (IQR, Z-score)
- Confidence interval estimation
- Robust measurement in noisy environments
- Method reliability assessment

**What you'll do**:

1. Measure same parameter using multiple methods (frequency, amplitude)
2. Apply majority voting to select most reliable result
3. Use weighted averaging based on method confidence
4. Detect and reject outlier measurements
5. Calculate confidence intervals for ensemble results
6. Validate robustness in noisy conditions

**Ensemble techniques**:

- **Majority voting** - Select most common result
- **Weighted averaging** - Combine results by confidence weights
- **Outlier rejection** - Remove statistically improbable results
- **Confidence intervals** - Estimate measurement uncertainty
- **Method scoring** - Assess reliability of each method

**Example workflow**:

```python
# Measure frequency using 3 methods
freq_fft = measure_frequency_fft(trace)      # FFT-based
freq_zerocross = measure_frequency_zc(trace) # Zero-crossing
freq_autocorr = measure_frequency_ac(trace)  # Autocorrelation

# Ensemble: majority voting
results = [freq_fft, freq_zerocross, freq_autocorr]
freq_final = majority_vote(results)

# Or: weighted average
weights = [0.5, 0.3, 0.2]  # FFT most reliable
freq_final = weighted_average(results, weights)
```

**Why this matters**: Single measurement methods can fail in noisy environments or edge cases. Ensemble methods provide robustness by combining multiple approaches and rejecting outliers.

---

### Demo 02: Signal Quality Scoring

**File**: `02_quality_scoring.py`

**What it teaches**:

- Multi-factor quality scoring (0-100 scale)
- SNR (Signal-to-Noise Ratio) estimation
- Distortion analysis (THD, SINAD)
- Noise floor analysis
- Automatic quality warnings
- Quality-based processing decisions

**What you'll do**:

1. Calculate comprehensive quality score for signals
2. Estimate SNR using multiple methods
3. Analyze harmonic distortion (THD)
4. Measure noise floor and dynamic range
5. Generate automatic quality warnings
6. Make processing decisions based on quality

**Quality metrics**:

| Metric        | Range | Good      | Acceptable      | Poor      |
| ------------- | ----- | --------- | --------------- | --------- |
| Overall Score | 0-100 | 90+       | 70-90           | <70       |
| SNR           | dB    | >60 dB    | 40-60 dB        | <40 dB    |
| THD           | %     | <1%       | 1-5%            | >5%       |
| Noise Floor   | dBFS  | <-80 dBFS | -60 to -80 dBFS | >-60 dBFS |

**Quality scoring algorithm**:

```python
def calculate_quality_score(trace):
    # Component scores (0-100)
    snr_score = score_snr(estimate_snr(trace))
    distortion_score = score_distortion(calculate_thd(trace))
    noise_score = score_noise(measure_noise_floor(trace))
    clipping_score = score_clipping(detect_clipping(trace))

    # Weighted average
    quality = (
        0.4 * snr_score +
        0.3 * distortion_score +
        0.2 * noise_score +
        0.1 * clipping_score
    )

    return quality
```

**Why this matters**: Quality scoring enables automated pass/fail decisions, prioritization of good data, and early detection of measurement problems. Essential for production testing and data validation.

---

### Demo 03: Automatic Warning System

**File**: `03_warning_system.py`

**What it teaches**:

- Automatic anomaly detection in signals
- Configurable warning thresholds
- Warning categories and severity levels
- Warning aggregation and reporting
- Context-aware warnings
- Warning suppression and filtering

**What you'll do**:

1. Detect various signal anomalies automatically
2. Configure warning thresholds for different scenarios
3. Categorize warnings by type and severity
4. Generate comprehensive warning reports
5. Filter warnings by severity and category
6. Implement warning suppression for known issues

**Warning categories**:

- **Signal quality** - SNR, clipping, saturation
- **Data integrity** - Missing samples, discontinuities
- **Measurement reliability** - Insufficient data, edge cases
- **Protocol errors** - Timing violations, invalid states
- **Configuration issues** - Sample rate mismatch, scale errors

**Severity levels**:

| Level    | Meaning           | Action Required         |
| -------- | ----------------- | ----------------------- |
| INFO     | Informational     | None - just FYI         |
| WARNING  | Potential issue   | Review recommended      |
| ERROR    | Confirmed problem | Investigation required  |
| CRITICAL | Severe problem    | Immediate action needed |

**Example output**:

```
Signal Quality Report
=====================

CRITICAL (1):
- Clipping detected at 234 samples (2.34%)
  Action: Reduce input amplitude or increase vertical scale

ERROR (2):
- SNR below threshold: 32 dB (minimum: 40 dB)
  Action: Improve signal quality or reduce noise
- Missing samples detected: 15 gaps
  Action: Check acquisition hardware

WARNING (3):
- Sample rate lower than recommended for frequency content
  Recommendation: Increase sample rate to 10x signal bandwidth
```

**Why this matters**: Automatic warnings catch quality issues before they invalidate analysis, saving time and preventing incorrect conclusions. Critical for automated testing and production validation.

---

### Demo 04: Smart Measurement Recommendations

**File**: `04_recommendations.py`

**What it teaches**:

- Context-aware measurement suggestions
- Parameter optimization guidance
- Best practice recommendations
- Troubleshooting assistance
- Analysis strategy suggestions
- Performance optimization tips

**What you'll do**:

1. Analyze signal characteristics automatically
2. Suggest appropriate measurements based on signal type
3. Recommend optimal analysis parameters
4. Provide best practice guidance
5. Assist with troubleshooting common issues
6. Suggest performance optimizations

**Recommendation types**:

**1. Measurement suggestions**:

```
Signal Type: Sine wave detected (1.2 kHz)
Recommended Measurements:
- Frequency (FFT-based for accuracy)
- RMS voltage
- THD (harmonic distortion)
- Phase analysis (if reference available)
```

**2. Parameter optimization**:

```
Current Settings:
- FFT size: 1024 points
- Window: Rectangular

Recommendations:
- Increase FFT size to 4096 for better frequency resolution
- Use Hanning window to reduce spectral leakage
- Expected improvement: 4x frequency resolution, -40 dB sidelobe reduction
```

**3. Best practices**:

```
Analysis: Power measurement
Best Practices:
✓ Use RMS for AC power (not peak)
✓ Ensure measurement window covers integer cycles
✓ Remove DC offset before calculation
✓ Verify load impedance for accurate power
```

**4. Troubleshooting**:

```
Issue: Low SNR (28 dB)
Possible Causes:
1. Input signal too weak → Increase signal amplitude
2. Noise on input → Add filtering or shielding
3. Quantization noise → Use higher resolution ADC
4. Ground loops → Check grounding

Diagnostic Steps:
1. Measure signal with scope to verify amplitude
2. Check for 50/60 Hz power line noise (FFT)
3. Verify ADC effective number of bits (ENOB)
```

**Why this matters**: Smart recommendations guide users toward correct analysis approaches, prevent common mistakes, and accelerate troubleshooting. Especially valuable for users learning reverse engineering or working with unfamiliar signal types.

---

## Running the Demonstrations

See [main demonstrations README](../README.md#running-demonstrations) for all execution options.

**Category-specific tip:** Start with the first demonstration (e.g., `01_ensemble_methods.py`) before exploring advanced examples.

---

## What You'll Learn

After completing this section, you will understand:

### Ensemble Methods

- **Multiple measurement approaches** - Frequency, amplitude, power via different methods
- **Voting algorithms** - Majority voting, weighted averaging
- **Outlier detection** - IQR method, Z-score method, modified Z-score
- **Confidence estimation** - Statistical uncertainty quantification
- **Robustness** - Reliable measurements in noisy conditions

### Quality Scoring

- **SNR estimation** - Multiple SNR calculation methods
- **Distortion analysis** - THD, SINAD, harmonic analysis
- **Noise characterization** - Noise floor, dynamic range, ENOB
- **Composite scoring** - Multi-factor quality assessment
- **Thresholds** - Application-specific quality criteria

### Warning Systems

- **Anomaly detection** - Clipping, saturation, discontinuities
- **Threshold configuration** - Context-specific warning levels
- **Severity classification** - INFO, WARNING, ERROR, CRITICAL
- **Warning aggregation** - Comprehensive reports
- **Filtering** - Suppress known issues, focus on new problems

### Smart Recommendations

- **Signal characterization** - Automatic signal type detection
- **Measurement selection** - Appropriate measurements for signal type
- **Parameter optimization** - FFT size, window functions, sample rate
- **Best practices** - Industry-standard analysis approaches
- **Troubleshooting** - Diagnostic guidance for common issues

---

## Common Issues and Solutions

### Ensemble methods giving inconsistent results

**Solution**: Ensure all methods use compatible parameters:

```python
# Problem: different FFT sizes
freq_method1 = measure_freq_fft(trace, fft_size=1024)
freq_method2 = measure_freq_fft(trace, fft_size=4096)  # Different!

# Solution: use same parameters
fft_size = 2048
freq_method1 = measure_freq_fft(trace, fft_size=fft_size)
freq_method2 = measure_freq_zerocrossing(trace)
freq_method3 = measure_freq_autocorr(trace)

# Now ensemble is meaningful
freq_ensemble = majority_vote([freq_method1, freq_method2, freq_method3])
```

### Quality scores always low

**Solution**: Verify quality thresholds match your application:

```python
# Default thresholds may be too strict
default_snr_threshold = 60  # dB - good for lab equipment

# Adjust for your application
embedded_snr_threshold = 40  # dB - typical for embedded systems
industrial_snr_threshold = 35  # dB - noisy industrial environment

quality_score = calculate_quality(
    trace,
    snr_threshold=industrial_snr_threshold  # Match your environment
)
```

### Too many warnings generated

**Solution**: Filter warnings by severity or suppress known issues:

```python
# Filter to ERROR and CRITICAL only
warnings = detect_anomalies(trace)
critical_warnings = [w for w in warnings
                    if w.severity in ["ERROR", "CRITICAL"]]

# Suppress known issues
suppress_list = ["sample_rate_low"]  # Known limitation
filtered_warnings = [w for w in warnings
                    if w.category not in suppress_list]
```

### Recommendations not applicable

**Solution**: Provide context for more relevant recommendations:

```python
# Generic recommendations
recs = get_recommendations(trace)  # May not fit your use case

# Context-specific recommendations
recs = get_recommendations(
    trace,
    application="automotive",  # CAN bus analysis
    constraints={"sample_rate": "fixed", "hardware": "embedded"},
    priority="accuracy"  # vs "speed"
)
```

---

## Next Steps: Where to Go After Quality Tools

### If You Want to...

| Goal                          | Next Demo                                   | Path                               |
| ----------------------------- | ------------------------------------------- | ---------------------------------- |
| Apply quality checks in batch | `09_batch_processing/01_parallel_batch.py`  | Batch → Quality validation         |
| Integrate quality in sessions | `10_sessions/01_analysis_session.py`        | Sessions → Quality tracking        |
| Build custom quality metrics  | `08_extensibility/02_custom_measurement.py` | Extensibility → Custom scoring     |
| Automate quality reports      | `15_export_visualization/`                  | Export → Quality dashboards        |
| Production quality gates      | `11_integration/`                           | Integration → CI/CD quality checks |

### Recommended Learning Sequence

1. **Complete Quality Tools** (this section)
   - Master quality assessment
   - Learn anomaly detection
   - Understand recommendations

2. **Apply to Batch Processing** (09_batch_processing/)
   - Quality scoring at scale
   - Aggregate quality metrics
   - Filter by quality thresholds

3. **Integrate with Sessions** (10_sessions/)
   - Track quality across recordings
   - Quality-based recording selection
   - Session quality reports

4. **Production Deployment** (11_integration/)
   - Automated quality gates
   - CI/CD quality checks
   - Production monitoring

5. **Custom Quality Tools** (08_extensibility/)
   - Domain-specific quality metrics
   - Custom warning rules
   - Application-specific recommendations

---

## Quality Tools Best Practices

**Ensemble Methods DO**: 3+ methods, weight by accuracy, document choices, validate

**Ensemble Methods DON'T**: Correlated methods, average without rejection, ignore confidence

**Quality Scoring DO**: Calibrate for application, multiple factors, document criteria, validate

**Quality Scoring DON'T**: Generic thresholds, single metric, skip noise floor

**Warning Systems DO**: Classify by severity, provide actionable advice, allow config, enable suppression

**Warning Systems DON'T**: Overwhelm users, vague messages, uniform severity

**Recommendations DO**: Specific advice, explain reasoning, prioritize impact, include examples

**Recommendations DON'T**: Generic advice, skip explanations, impossible actions, ignore constraints

## Quality Metrics Reference

**SNR Methods**: Frequency domain (tonal), time domain (broadband), autocorrelation (periodic), peak-to-RMS (simple)

**Distortion Metrics**: THD = harmonics/fundamental, THD+N = (noise² + distortion²)/signal, SINAD = signal/(noise+distortion), SFDR = fundamental/largest_spur

**Thresholds**: Lab (80 dB SNR, 0.1% THD), Industrial (50 dB, 2%), Automotive (40 dB, 5%), Embedded (35 dB, 10%)

---

## Resources

### In This Repository

- **`src/oscura/quality/`** - Quality assessment tools (future)
- **`tests/unit/test_quality.py`** - Quality metric tests (future)
- **`examples/quality/`** - Quality assessment examples (future)

### Statistical Methods

- **[SciPy Statistics](https://docs.scipy.org/doc/scipy/reference/stats.html)** - Statistical functions
- **[NumPy Statistics](https://numpy.org/doc/stable/reference/routines.statistics.html)** - Basic statistics
- **[IEEE 1241-2010](https://standards.ieee.org/)** - ADC terminology and testing

### Signal Quality

- **SNR measurement** - IEEE 1057, IEEE 1241
- **THD analysis** - IEC 61000-4-7
- **Noise analysis** - IEEE 1057

---

## Summary

The Quality Tools section covers:

| Demo                | Focus                    | Outcome                          |
| ------------------- | ------------------------ | -------------------------------- |
| 01_ensemble_methods | Multiple methods         | Robust measurements              |
| 02_quality_scoring  | SNR, distortion, scoring | Comprehensive quality assessment |
| 03_warning_system   | Anomaly detection        | Automatic issue detection        |
| 04_recommendations  | Smart suggestions        | Analysis guidance                |

After completing these 60-minute demonstrations, you'll be able to:

- Assess signal quality comprehensively
- Detect anomalies and quality issues automatically
- Implement robust measurement strategies
- Generate smart analysis recommendations
- Build automated quality gates
- Troubleshoot measurement problems effectively

**Ready to start?** Run this to understand ensemble methods:

```bash
python demonstrations/12_quality_tools/01_ensemble_methods.py
```

Happy quality assurance!
