# Exploratory Analysis Techniques

**Advanced techniques for analyzing unknown and poorly-documented signals.**

This section contains 4 demonstrations showing how to characterize unknown signals, perform fuzzy pattern matching, recover corrupted signals, and conduct exploratory analysis when you have no documentation. Essential for reverse engineering unfamiliar hardware.

---

## Prerequisites

Before running these demonstrations, ensure you have:

- **Completed Basic Analysis** - Run `demonstrations/02_basic_analysis/` first
- **Understanding of FFT** - Familiarity with frequency domain analysis
- **Pattern Recognition Concepts** - Basic understanding of pattern matching
- **Statistical Knowledge** - Mean, standard deviation, correlation

Check your setup:

```bash
python demonstrations/02_basic_analysis/01_waveform_measurements.py
# Should show successful measurements
```

---

## Demonstrations

| Demo                    | Time       | Difficulty   | Focus                               |
| ----------------------- | ---------- | ------------ | ----------------------------------- |
| 01_unknown_signals      | 15 min     | Advanced     | Automatic signal characterization   |
| 02_fuzzy_matching       | 10 min     | Intermediate | Pattern matching with tolerance     |
| 03_signal_recovery      | 15 min     | Advanced     | Recovering corrupted signals        |
| 04_exploratory_analysis | 15 min     | Advanced     | Investigation without documentation |
| **Total**               | **55 min** | **Advanced** | **Unknown signal analysis**         |

---

## Learning Path

Complete these demonstrations in order for progressive skill building:

```
01_unknown_signals.py      02_fuzzy_matching.py     03_signal_recovery.py    04_exploratory_analysis.py
         ↓                         ↓                         ↓                          ↓
Characterize unknowns      Find similar patterns    Recover corrupted data   Complete investigation
Type detection             Fuzzy comparison         Noise removal            No-documentation RE
```

### Estimated Time: 55 minutes

---

## Key Concepts

This section teaches:

1. **Unknown Signal Characterization** - Automatic detection and classification
2. **Fuzzy Pattern Matching** - Finding patterns with tolerance for variations
3. **Signal Recovery** - Reconstructing signals from noisy or corrupted data
4. **Exploratory Investigation** - Systematic reverse engineering without docs
5. **Feature Extraction** - Identifying key signal characteristics

---

## Running Demonstrations

### Option 1: Run Individual Demo

```bash
# From the project root
python demonstrations/14_exploratory/01_unknown_signals.py

# Or from the demo directory
cd demonstrations/14_exploratory
python 01_unknown_signals.py
```

### Option 2: Run All Exploratory Demos

```bash
# From the project root
python demonstrations/14_exploratory/01_unknown_signals.py && \
python demonstrations/14_exploratory/02_fuzzy_matching.py && \
python demonstrations/14_exploratory/03_signal_recovery.py && \
python demonstrations/14_exploratory/04_exploratory_analysis.py
```

### Option 3: Validate All Demonstrations

```bash
# From the project root
python demonstrations/validate_all.py
```

---

## What You'll Learn

### Demo 01: Unknown Signal Characterization

**File**: `01_unknown_signals.py`

**Demonstrates**:

- Automatic signal type detection (analog/digital/mixed)
- Frequency and pattern detection
- Modulation detection (AM, FM, PSK, etc.)
- Signal classification and feature extraction

**What you'll do**:

1. Analyze completely unknown signals
2. Automatically detect signal type and characteristics
3. Identify potential protocols from signal patterns
4. Extract key features for further investigation

**Capabilities**:

- `oscura.exploratory.signal_detection` - Auto signal type detection
- `oscura.exploratory.type_classification` - Signal classification
- `oscura.exploratory.pattern_recognition` - Pattern identification
- `oscura.exploratory.feature_extraction` - Key feature extraction

**Related Demos**:

- `14_exploratory/02_fuzzy_matching.py` - Pattern matching
- `03_protocol_decoding/06_auto_detection.py` - Protocol auto-detection
- `13_guidance/01_smart_recommendations.py` - Guided analysis

---

### Demo 02: Fuzzy Pattern Matching

**File**: `02_fuzzy_matching.py`

**Demonstrates**:

- Fuzzy pattern matching with configurable tolerance
- Template matching with noise robustness
- Similarity metrics (correlation, DTW, cosine)
- Pattern discovery in noisy signals

**What you'll do**:

1. Search for patterns in signals with fuzzy matching
2. Configure tolerance thresholds for different noise levels
3. Use multiple similarity metrics for robust matching
4. Discover repeated patterns in unknown signals

**Capabilities**:

- Fuzzy template matching
- Multiple similarity metrics
- Noise-robust pattern detection
- Automatic tolerance tuning

**Related Demos**:

- `14_exploratory/01_unknown_signals.py` - Signal characterization
- `06_reverse_engineering/02_pattern_analysis.py` - Pattern analysis
- `14_exploratory/03_signal_recovery.py` - Signal recovery

---

### Demo 03: Signal Recovery

**File**: `03_signal_recovery.py`

**Demonstrates**:

- Recovering signals from noisy data
- Corrupted data reconstruction
- Missing sample interpolation
- Denoising techniques (filtering, averaging, smoothing)

**What you'll do**:

1. Recover signals corrupted by noise
2. Reconstruct signals with missing samples
3. Apply advanced denoising techniques
4. Validate recovered signal quality

**Capabilities**:

- Noise removal and filtering
- Missing data interpolation
- Signal reconstruction
- Quality validation

**Related Demos**:

- `02_basic_analysis/03_filtering.py` - Filtering techniques
- `14_exploratory/02_fuzzy_matching.py` - Fuzzy matching
- `04_advanced_analysis/01_spectral_analysis.py` - Frequency analysis

---

### Demo 04: Exploratory Analysis

**File**: `04_exploratory_analysis.py`

**Demonstrates**:

- Systematic reverse engineering without documentation
- Hypothesis-driven investigation
- Multi-method analysis for validation
- Building signal understanding from scratch

**What you'll do**:

1. Perform complete exploratory analysis on unknown signals
2. Use systematic investigation techniques
3. Validate hypotheses with multiple analysis methods
4. Build comprehensive signal documentation from findings

**Capabilities**:

- Systematic investigation workflows
- Multi-method validation
- Hypothesis testing
- Documentation generation

**Related Demos**:

- `14_exploratory/01_unknown_signals.py` - Signal characterization
- `06_reverse_engineering/01_unknown_protocol.py` - Protocol RE
- `16_complete_workflows/01_protocol_discovery.py` - Complete workflows

---

## Troubleshooting

### "Cannot detect signal type"

**Solution**: Signal may be too noisy or too short. Try:

```python
# Ensure adequate signal length
assert len(trace.data) >= 1000  # Minimum samples for detection

# Apply noise reduction first
from oscura.analyzers.filters import lowpass_filter
filtered = lowpass_filter(trace, cutoff=10000.0)
```

### "Pattern matching returns no results"

**Solution**: Tolerance may be too strict. Adjust fuzzy matching threshold:

```python
# Increase tolerance for noisy signals
matches = fuzzy_match(signal, pattern, tolerance=0.2)  # 20% tolerance

# Try different similarity metrics
matches = fuzzy_match(signal, pattern, metric="dtw")  # Dynamic time warping
```

### "Signal recovery produces artifacts"

**Solution**: Recovery parameters may need tuning:

```python
# Use gentler filtering
recovered = recover_signal(corrupted, filter_strength=0.3)

# Try different interpolation methods
recovered = recover_signal(corrupted, interpolation="cubic")
```

---

## Next Steps

### If You Want to...

| Goal                          | Next Demo                                         | Path                |
| ----------------------------- | ------------------------------------------------- | ------------------- |
| Decode unknown protocols      | `06_reverse_engineering/01_unknown_protocol.py`   | Reverse engineering |
| Export findings for reporting | `15_export_visualization/04_report_generation.py` | Report generation   |
| Build complete RE workflow    | `16_complete_workflows/01_protocol_discovery.py`  | Complete workflows  |
| Compare against references    | `18_comparison_testing/01_golden_reference.py`    | Comparison testing  |

### Recommended Next Sections

1. **Reverse Engineering** (06_reverse_engineering/)
   - Unknown protocol decoding
   - Pattern analysis
   - State machine extraction

2. **Complete Workflows** (16_complete_workflows/)
   - End-to-end reverse engineering
   - Production workflows
   - Real-world case studies

3. **Export and Visualization** (15_export_visualization/)
   - Document findings
   - Create visualizations
   - Generate reports

---

## Understanding Exploratory Analysis

### Signal Characterization Pipeline

Oscura's exploratory analysis follows this pipeline:

1. **Initial Detection** - Identify signal type (analog/digital/mixed)
2. **Feature Extraction** - Extract key characteristics (frequency, amplitude, patterns)
3. **Classification** - Categorize signal based on features
4. **Pattern Recognition** - Identify repeated structures
5. **Hypothesis Testing** - Validate findings with multiple methods

### Fuzzy Matching Techniques

Oscura supports multiple similarity metrics:

- **Correlation** - Statistical similarity (fast, good for aligned signals)
- **DTW** - Dynamic Time Warping (handles timing variations)
- **Cosine Similarity** - Angle-based comparison (robust to amplitude changes)
- **Euclidean Distance** - Point-to-point difference (simple, intuitive)

### Signal Recovery Methods

Recovery techniques include:

- **Filtering** - Remove frequency-based noise (lowpass, highpass, bandpass)
- **Smoothing** - Reduce jitter (moving average, Savitzky-Golay)
- **Interpolation** - Fill missing samples (linear, cubic, spline)
- **Deconvolution** - Remove known distortions (Wiener filter)

---

## Best Practices

### Unknown Signal Investigation

1. **Start Broad** - Use automatic characterization first
2. **Narrow Down** - Focus on specific characteristics found
3. **Validate** - Use multiple methods to confirm findings
4. **Document** - Record all findings and hypotheses

### Fuzzy Matching Strategy

1. **Start Strict** - Begin with low tolerance (0.05-0.10)
2. **Increase Gradually** - Raise tolerance if no matches found
3. **Try Multiple Metrics** - Different metrics work for different signals
4. **Validate Matches** - Check that matches make sense in context

### Signal Recovery Approach

1. **Characterize Corruption** - Understand noise type and extent
2. **Apply Minimal Processing** - Start with gentlest recovery methods
3. **Validate Quality** - Compare recovered signal to expectations
4. **Iterate if Needed** - Apply stronger methods if initial recovery insufficient

---

## Advanced Techniques

### Multi-Method Validation

Use multiple analysis methods to validate findings:

```python
# Frequency domain analysis
fft_result = fft_analysis(trace)

# Time domain analysis
time_result = waveform_analysis(trace)

# Statistical analysis
stats_result = statistical_analysis(trace)

# Validate consistency across methods
assert all_methods_agree(fft_result, time_result, stats_result)
```

### Hypothesis-Driven Investigation

Systematic approach to unknown signals:

```python
# 1. Form hypothesis based on initial characterization
hypothesis = "Signal contains 9600 baud UART data"

# 2. Test hypothesis with specific analysis
uart_test = uart_decode(trace, baud_rate=9600)

# 3. Validate or refine hypothesis
if uart_test.success:
    # Hypothesis confirmed
    analyze_uart_data(uart_test.packets)
else:
    # Refine hypothesis
    hypothesis = "Signal may be different baud rate"
    test_multiple_baud_rates(trace)
```

### Iterative Refinement

Build understanding incrementally:

```python
# Iteration 1: Basic characterization
info_v1 = characterize_signal(trace)

# Iteration 2: Refine based on initial findings
if info_v1.type == "digital":
    info_v2 = characterize_digital_protocol(trace)

# Iteration 3: Decode specific protocol
if info_v2.likely_protocol == "UART":
    decoded = decode_uart(trace, info_v2.parameters)
```

---

## Tips for Success

### Maximize Detection Accuracy

Provide clean signals for best results:

```python
# Remove DC offset first
trace_normalized = remove_dc_offset(trace)

# Apply appropriate filtering
trace_clean = filter_signal(trace_normalized, cutoff=50000.0)

# Now characterize
characteristics = characterize_signal(trace_clean)
```

### Choose Right Similarity Metric

Different metrics for different scenarios:

- **Aligned signals** → Use correlation (fastest)
- **Timing variations** → Use DTW (handles stretching)
- **Amplitude variations** → Use cosine similarity (scale-invariant)
- **Unknown relationship** → Try all three and compare

### Optimize Recovery Quality

Balance noise removal with signal preservation:

```python
# Too aggressive - may remove signal features
over_filtered = recover(signal, strength=0.9)

# Too gentle - may not remove enough noise
under_filtered = recover(signal, strength=0.1)

# Just right - removes noise, preserves signal
optimal = recover(signal, strength=0.4)
```

---

## Summary

The Exploratory Analysis section covers:

| Demo                    | Focus                      | Outcome                             |
| ----------------------- | -------------------------- | ----------------------------------- |
| 01_unknown_signals      | Automatic characterization | Signal type and feature detection   |
| 02_fuzzy_matching       | Pattern matching           | Find patterns with tolerance        |
| 03_signal_recovery      | Corruption recovery        | Reconstruct signals from noisy data |
| 04_exploratory_analysis | Complete investigation     | Systematic no-documentation RE      |

After completing these 55-minute demonstrations, you'll understand:

- How to characterize completely unknown signals
- How to find patterns in noisy data with fuzzy matching
- How to recover signals from corruption and noise
- How to perform systematic reverse engineering without documentation
- How to validate findings with multiple analysis methods

**Ready to start?** Run this to characterize unknown signals:

```bash
python demonstrations/14_exploratory/01_unknown_signals.py
```

Happy exploring the unknown!
