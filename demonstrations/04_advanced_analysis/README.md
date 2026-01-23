# Advanced Analysis

**Master advanced signal processing techniques for jitter, power, signal integrity, and pattern discovery.**

This section contains 6 demonstrations covering IEEE 2414-2020 jitter analysis, IEEE 1459-2010 power quality, signal integrity metrics, eye diagrams, pattern discovery, and comprehensive quality assessment. Perfect for high-speed design validation and reverse engineering complex systems.

---

## Prerequisites

Before running these demonstrations, ensure you have:

- **Completed Getting Started** - Finish `00_getting_started/` first
- **Completed Basic Analysis** - Finish `02_basic_analysis/` REQUIRED
- **Python 3.12+** - Oscura requires Python 3.12 or higher
- **Oscura installed** - Install with `pip install oscura` or `uv add oscura`
- **Advanced signal processing knowledge** - Understanding jitter, power quality, SI concepts
- **IEEE standards familiarity** - Knowledge of IEEE 2414, 1459, 181 helpful

---

## Demonstrations

| Demo      | File                       | Time        | Difficulty                   | Topics                                             |
| --------- | -------------------------- | ----------- | ---------------------------- | -------------------------------------------------- |
| **01**    | `01_jitter_analysis.py`    | 20 min      | Advanced                     | TIE, C2C jitter, DCD, RJ/DJ decomposition          |
| **02**    | `02_power_analysis.py`     | 15 min      | Intermediate                 | Active/reactive/apparent power, PF, THD            |
| **03**    | `03_signal_integrity.py`   | 20 min      | Advanced                     | Rise/fall time, overshoot, eye metrics, TDR        |
| **04**    | `04_eye_diagrams.py`       | 20 min      | Advanced                     | Eye construction, height/width, Q-factor, BER      |
| **05**    | `05_pattern_discovery.py`  | 20 min      | Advanced                     | Signature discovery, repetitions, state extraction |
| **06**    | `06_quality_assessment.py` | 15 min      | Intermediate                 | SNR, SINAD, THD, SFDR, ENOB                        |
| **Total** |                            | **110 min** | **Intermediate to Advanced** | **Professional-grade analysis**                    |

---

## Learning Path

These demonstrations are designed to be completed **in order**. Each builds on concepts from the previous one:

```
01_jitter_analysis.py → 02_power_analysis.py → 03_signal_integrity.py
        ↓                       ↓                          ↓
  Timing analysis         Power quality              Signal quality
  TIE, C2C, period        P, Q, S, PF, THD          Rise/fall, overshoot
  RJ/DJ, bathtub          Phase angle, ripple       Eye metrics, TDR
        ↓                       ↓                          ↓
04_eye_diagrams.py → 05_pattern_discovery.py → 06_quality_assessment.py
        ↓                    ↓                          ↓
  BER analysis          Pattern recognition       Quality scoring
  Eye height/width      Signatures, repetitions   SNR, SINAD, ENOB
  Q-factor, crossing    State extraction          SFDR, warnings
```

### Recommended Time

**Intermediate path** (50 min): Demos 02, 03, 06
**Advanced path** (90 min): Demos 01, 03, 04, 06
**Expert path** (110 min): All demos

---

## Key Concepts

### What You'll Learn

**Jitter Analysis** (Demo 01):

- Time Interval Error (TIE) measurement
- Cycle-to-cycle jitter (C2C)
- Period jitter analysis
- Duty cycle distortion (DCD)
- Random vs deterministic jitter (RJ/DJ) decomposition
- Bathtub curves for BER vs threshold analysis

**Power Analysis** (Demo 02):

- Active power (P) in watts
- Reactive power (Q) in VAR
- Apparent power (S) in VA
- Power factor (PF = P/S)
- Phase angle between voltage and current
- Total harmonic distortion for power (THD-P)
- DC-DC converter ripple analysis
- Power conversion efficiency calculations

**Signal Integrity** (Demo 03):

- Rise time and fall time (10-90%)
- Overshoot and undershoot measurements
- Comprehensive SI report generation
- Eye diagram quality metrics
- Time-domain reflectometry (TDR) for impedance profiling
- High-speed link quality assessment

**Eye Diagrams** (Demo 04):

- Eye diagram construction from serial data
- Eye height (vertical opening) measurement
- Eye width (horizontal opening) measurement
- Q-factor (signal quality metric)
- Crossing percentage (duty cycle distortion indicator)
- BER contours and probability analysis
- Comprehensive eye metrics extraction

**Pattern Discovery** (Demo 05):

- Automatic header/delimiter discovery
- Periodic pattern and repetition detection
- Sequence identification and extraction
- Pattern correlation analysis
- Digital state machine inference
- Unknown protocol fingerprinting

**Quality Assessment** (Demo 06):

- Signal-to-noise ratio (SNR)
- Signal-to-noise and distortion (SINAD)
- Total harmonic distortion (THD)
- Spurious-free dynamic range (SFDR)
- Effective number of bits (ENOB)
- Overall signal quality scoring
- Automatic data quality warnings

---

## Running Demonstrations

### Option 1: Run Individual Demo

Run a single demo to focus on a specific technique:

```bash
# From the project root
python demonstrations/04_advanced_analysis/01_jitter_analysis.py

# Or from the demo directory
cd demonstrations/04_advanced_analysis
python 01_jitter_analysis.py
```

Expected output: IEEE-compliant metrics with validation.

### Option 2: Run All Advanced Analysis Demos

Run all six demos in sequence:

```bash
# From the project root
for demo in demonstrations/04_advanced_analysis/*.py; do
    python "$demo"
done
```

### Option 3: Validate All Demonstrations

Validate all demonstrations in the project:

```bash
# From the project root
python demonstrations/validate_all.py
```

This runs all demonstrations and reports coverage, validation status, and failures.

---

## What You'll Learn

### IEEE Standard Compliance

**IEEE 2414-2020** (Jitter and Phase Noise):

- Time Interval Error (TIE) definition
- Cycle-to-cycle jitter measurement
- Period jitter calculation
- RJ/DJ decomposition methodology
- Bathtub curve construction

**IEEE 1459-2010** (Power Quality):

- Active power measurement
- Reactive power calculation
- Apparent power definition
- Power factor computation
- Harmonic power distortion

**IEEE 1241-2010** (ADC Testing):

- SNR measurement methodology
- SINAD calculation from spectrum
- THD from harmonic components
- SFDR from spurious peaks
- ENOB derived from SINAD

**IEEE 181-2011** (Waveform Measurements):

- Rise time and fall time (10-90%)
- Overshoot and undershoot
- Pulse parameter definitions
- Transitional waveform terminology

### Advanced Techniques

**Jitter Characterization**:

- Total jitter (TJ) measurement
- Random jitter (RJ) using histogram method
- Deterministic jitter (DJ) extraction
- Dual-Dirac model for RJ
- BER extrapolation from measured jitter

**Power Quality Analysis**:

- Three-phase power calculations
- Harmonic analysis and limits
- Interharmonics and subharmonics
- Power factor correction
- Efficiency measurement for converters

**Signal Integrity Assessment**:

- Transmission line effects
- Impedance discontinuities via TDR
- Crosstalk analysis
- PCB design quality metrics
- High-speed link margin analysis

**Eye Diagram Analysis**:

- Multi-cycle overlay construction
- Statistical eye generation
- Mask testing and compliance
- Vertical and horizontal margins
- Q-factor to BER conversion

**Pattern Recognition**:

- Entropy-based signature discovery
- Cross-correlation for repetition
- State machine extraction
- Protocol structure inference
- Anomaly detection

**Quality Metrics**:

- Multi-dimensional quality scoring
- Automatic issue flagging
- Noise floor characterization
- Dynamic range assessment
- Confidence intervals

---

## Common Issues and Solutions

### "Jitter measurement shows unexpected results"

**Solution**: Jitter analysis requires clean clock edges:

1. Ensure sufficient edge transitions (100+ edges minimum)
2. Check for noise affecting edge detection
3. Validate threshold settings for edge detection
4. Consider filtering before jitter analysis
5. Verify clock is reasonably stable

### "Power factor calculation shows values > 1.0"

**Solution**: Measurement window or phase alignment issues:

1. Ensure voltage and current traces are time-aligned
2. Check that measurement includes integer number of cycles
3. Validate phase angle calculation
4. Look for harmonics affecting measurements
5. Verify reactive power sign convention

### "Eye diagram appears distorted"

**Solution**: Synchronization or unit interval issues:

1. Verify unit interval (bit time) is correct
2. Check for clock recovery accuracy
3. Ensure sufficient data edges for overlay
4. Look for significant jitter affecting alignment
5. Validate trigger/sync settings

### "Pattern discovery finds spurious patterns"

**Solution**: Noise or insufficient data:

1. Increase minimum pattern length threshold
2. Require higher repetition count for validation
3. Apply filtering to reduce noise
4. Use longer capture window for statistics
5. Adjust correlation threshold for matching

### "Quality metrics differ from oscilloscope"

**Solution**: Measurement method or windowing differences:

1. Check FFT window function (Hann, Hamming, etc.)
2. Verify coherent sampling (integer cycles in window)
3. Compare fundamental frequency identification
4. Validate harmonic peak detection
5. Check for different ENOB calculation methods

---

## Next Steps: Where to Go After Advanced Analysis

### If You Want to...

| Goal                            | Next Demo                                         | Path                             |
| ------------------------------- | ------------------------------------------------- | -------------------------------- |
| Apply to automotive diagnostics | `05_domain_specific/01_automotive_diagnostics.py` | Analysis → OBD-II/J1939          |
| EMC compliance testing          | `05_domain_specific/02_emc_compliance.py`         | Analysis → CISPR/MIL-STD         |
| Vintage logic family detection  | `05_domain_specific/03_vintage_logic.py`          | Analysis → IC identification     |
| Side-channel analysis           | `05_domain_specific/04_side_channel.py`           | Analysis → Cryptographic attacks |
| Build complete RE workflow      | `16_complete_workflows/01_protocol_discovery.py`  | Analysis → Full workflow         |

### Recommended Learning Sequence

1. **Master Advanced Analysis** (this section)
   - Understand jitter and timing quality
   - Learn power quality assessment
   - Apply signal integrity techniques

2. **Explore Domain-Specific Applications** (05_domain_specific/)
   - Real-world problem solving
   - Industry standards compliance
   - Specialized analysis workflows

3. **Complete Workflows** (16_complete_workflows/)
   - End-to-end reverse engineering
   - Multi-stage analysis pipelines
   - Production-ready solutions

---

## Tips for Learning

### Start with Quality Assessment

Demo 06 provides a comprehensive overview:

```python
from oscura import snr, sinad, thd, sfdr, enob

# Quick quality check
signal_snr = snr(trace, signal_freq=1000.0)
signal_thd = thd(trace, fundamental_freq=1000.0)
effective_bits = enob(trace, signal_freq=1000.0)

if signal_snr < 40:
    print("Warning: Low SNR - check for noise")
if signal_thd > 0.01:
    print("Warning: High THD - check for distortion")
```

### Understand IEEE Standards

Advanced analysis requires understanding standard definitions:

```python
# IEEE 2414-2020: TIE is edge time minus ideal edge time
tie = edge_times - ideal_edge_times

# IEEE 1459-2010: Apparent power S = V_rms * I_rms
apparent_power = v_rms * i_rms

# IEEE 1241-2010: ENOB = (SINAD - 1.76) / 6.02
enob = (sinad_db - 1.76) / 6.02
```

Read the standards for authoritative methodology.

### Visualize Advanced Metrics

Plotting helps understand complex measurements:

```python
import matplotlib.pyplot as plt

# Plot jitter histogram
plt.hist(tie_values, bins=50)
plt.xlabel("Time Interval Error (s)")
plt.ylabel("Count")
plt.title("TIE Histogram - RJ/DJ Decomposition")

# Plot eye diagram
eye = generate_eye(trace, unit_interval=1e-9)
plt.imshow(eye.density, aspect='auto', origin='lower')
plt.xlabel("Time (UI)")
plt.ylabel("Voltage")
plt.title(f"Eye Diagram - Q={eye.q_factor:.2f}")
```

### Combine Multiple Techniques

Real-world analysis uses multiple methods:

```python
# Complete signal quality workflow

# 1. Time domain quality
snr_val = snr(trace, signal_freq=1000.0)
thd_val = thd(trace, fundamental_freq=1000.0)

# 2. Jitter analysis
edges = find_rising_edges(trace, threshold=0.5)
tie_result = tie_from_edges(edges, expected_period=1e-3)

# 3. Eye diagram for BER
eye = generate_eye(trace, unit_interval=1e-9)
eye_metrics = measure_eye(eye)

# 4. Overall assessment
if snr_val > 40 and thd_val < 0.01 and eye_metrics.q_factor > 6:
    print("Signal quality: EXCELLENT")
```

### Reference Professional Tools

Compare Oscura results with industry tools:

- **Oscilloscope jitter analysis** - Compare TIE, C2C measurements
- **Power analyzer** - Validate P, Q, S, PF calculations
- **BERT tester** - Compare eye diagrams and BER
- **Spectrum analyzer** - Verify SNR, SFDR, ENOB

---

## Understanding the Framework

### Jitter Analysis API

```python
from oscura.analyzers.jitter.measurements import (
    tie_from_edges,
    cycle_to_cycle_jitter,
    period_jitter,
    measure_dcd
)

# Time Interval Error
edges = find_rising_edges(trace, threshold=0.5)
tie_result = tie_from_edges(edges, expected_period=1e-3)

# Cycle-to-cycle jitter
c2c_result = cycle_to_cycle_jitter(edges)

# Period jitter
pj_result = period_jitter(edges, nominal_period=1e-3)

# Duty cycle distortion
dcd_result = measure_dcd(trace, threshold=0.5)
```

### Power Analysis API

```python
from oscura.analyzers.power.ac_power import (
    apparent_power,
    reactive_power,
    power_factor,
    phase_angle
)
from oscura.analyzers.power.basic import average_power

# AC power analysis
p_active = average_power(v_trace, i_trace)
q_reactive = reactive_power(v_trace, i_trace)
s_apparent = apparent_power(v_trace, i_trace)
pf = power_factor(v_trace, i_trace)
phi = phase_angle(v_trace, i_trace)
```

### Signal Integrity API

```python
from oscura import rise_time, fall_time, overshoot, undershoot

# Transition measurements
rt = rise_time(trace, low_threshold=0.1, high_threshold=0.9)
ft = fall_time(trace, low_threshold=0.1, high_threshold=0.9)

# Overshoot/undershoot
ovs = overshoot(trace)
uds = undershoot(trace)
```

### Eye Diagram API

```python
from oscura.analyzers.eye.diagram import generate_eye
from oscura.analyzers.eye.metrics import measure_eye, eye_height, eye_width, q_factor

# Generate eye diagram
eye = generate_eye(trace, unit_interval=1e-9, samples_per_ui=100)

# Extract metrics
metrics = measure_eye(eye)
height = eye_height(eye)
width = eye_width(eye)
q = q_factor(eye)
```

### Pattern Discovery API

```python
from oscura.analyzers.patterns.discovery import SignatureDiscovery

# Discover patterns
discovery = SignatureDiscovery(min_length=4, max_length=32)
signatures = discovery.discover_signatures(trace.data)

for sig in signatures:
    print(f"Pattern: {sig.pattern.hex()} (count: {sig.count})")
```

---

## Resources

### In This Repository

- **`src/oscura/analyzers/jitter/`** - Jitter analysis implementations
- **`src/oscura/analyzers/power/`** - Power quality analysis
- **`src/oscura/analyzers/eye/`** - Eye diagram generation and metrics
- **`src/oscura/analyzers/patterns/`** - Pattern discovery algorithms
- **`tests/unit/analyzers/`** - Advanced analysis test cases

### External Resources

- **[IEEE 2414-2020](https://standards.ieee.org/)** - Jitter and phase noise standard
- **[IEEE 1459-2010](https://standards.ieee.org/)** - Power quality definitions
- **[IEEE 1241-2010](https://standards.ieee.org/)** - ADC testing standard
- **[JEDEC JESD65B](https://www.jedec.org/)** - Eye diagram measurement
- **[Tektronix Jitter Analysis](https://www.tek.com/)** - Application notes

### Professional Tools

Compare with industry-standard tools:

- **Tektronix DPOJET** - Jitter and eye diagram analysis
- **Keysight Infiniium** - Advanced signal integrity
- **LeCroy WavePro** - Power analysis and harmonics
- **Teledyne LeCroy** - Eye diagram and mask testing

### Getting Help

1. Reference IEEE standards for measurement definitions
2. Check demo docstrings for detailed methodology
3. Review test cases in `tests/unit/analyzers/`
4. Compare with professional tool results
5. Validate with known reference signals

---

## Summary

The Advanced Analysis section covers:

| Demo                  | Focus               | Outcome                                     |
| --------------------- | ------------------- | ------------------------------------------- |
| 01_jitter_analysis    | Timing quality      | TIE, C2C, period jitter, RJ/DJ, bathtub     |
| 02_power_analysis     | Power quality       | P, Q, S, PF, phase angle, THD, efficiency   |
| 03_signal_integrity   | Signal quality      | Rise/fall time, overshoot, eye metrics, TDR |
| 04_eye_diagrams       | BER analysis        | Eye height/width, Q-factor, BER contours    |
| 05_pattern_discovery  | Pattern recognition | Signatures, repetitions, state extraction   |
| 06_quality_assessment | Quality scoring     | SNR, SINAD, THD, SFDR, ENOB                 |

After completing these six 110-minute demonstrations, you'll understand:

- IEEE-compliant jitter and timing analysis
- Power quality assessment and efficiency
- High-speed signal integrity techniques
- Eye diagram generation and BER prediction
- Automatic pattern discovery and recognition
- Comprehensive signal quality assessment

**Ready to start?** Run this to begin with quality assessment:

```bash
python demonstrations/04_advanced_analysis/06_quality_assessment.py
```

Happy analyzing!
