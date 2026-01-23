# Basic Analysis

**Master fundamental signal measurements and analysis techniques for hardware reverse engineering.**

This section contains 6 demonstrations covering waveform measurements, statistical analysis, spectral analysis, filtering, triggering, and mathematical operations. Learn the essential analysis techniques used in every reverse engineering workflow.

---

## Prerequisites

Before running these demonstrations, ensure you have:

- **Completed Getting Started** - Finish `00_getting_started/` first
- **Completed Data Loading** - Finish `01_data_loading/` recommended
- **Python 3.12+** - Oscura requires Python 3.12 or higher
- **Oscura installed** - Install with `pip install oscura` or `uv add oscura`
- **Basic signal processing knowledge** - Understanding time/frequency domain helps
- **NumPy/SciPy** - Automatically installed with Oscura

---

## Demonstrations

| Demo      | File                          | Time       | Difficulty                   | Topics                                                    |
| --------- | ----------------------------- | ---------- | ---------------------------- | --------------------------------------------------------- |
| **01**    | `01_waveform_measurements.py` | 15 min     | Beginner                     | Amplitude, frequency, rise/fall time, duty cycle          |
| **02**    | `02_statistics.py`            | 10 min     | Beginner                     | Mean, std, percentiles, distributions, outliers           |
| **03**    | `03_spectral_analysis.py`     | 15 min     | Intermediate                 | FFT, PSD, THD, SNR, SINAD, ENOB, SFDR                     |
| **04**    | `04_filtering.py`             | 15 min     | Intermediate                 | Low/high/band-pass, filter design, types                  |
| **05**    | `05_triggering.py`            | 15 min     | Intermediate                 | Edge detection, pulse width, glitches, runts              |
| **06**    | `06_math_operations.py`       | 10 min     | Beginner                     | Add, subtract, multiply, divide, differentiate, integrate |
| **Total** |                               | **80 min** | **Beginner to Intermediate** | **Complete analysis foundations**                         |

---

## Learning Path

These demonstrations are designed to be completed **in order**. Each builds on concepts from the previous one:

```
01_waveform_measurements.py → 02_statistics.py → 03_spectral_analysis.py
        ↓                            ↓                       ↓
  Time domain basics         Distribution analysis    Frequency domain
  Amplitude, frequency       Mean, std, outliers      FFT, harmonics, noise
  Rise/fall, duty cycle      Percentiles, skewness    THD, SNR, ENOB
        ↓                            ↓                       ↓
04_filtering.py → 05_triggering.py → 06_math_operations.py
        ↓                ↓                      ↓
  Signal conditioning  Event detection    Signal manipulation
  Low/high/band-pass   Edges, pulses      Add, subtract, integrate
  Butterworth, Bessel  Glitches, runts    Correlation, FFT
```

### Recommended Time

**Beginner path** (40 min): Demos 01, 02, 06
**Intermediate path** (60 min): Demos 01-03, 05
**Advanced path** (80 min): All demos

---

## Key Concepts

### What You'll Learn

**Waveform Measurements** (Demo 01):

- Amplitude (peak-to-peak voltage)
- Frequency and period detection
- Rise time and fall time (10-90%)
- Duty cycle and pulse width
- Overshoot and undershoot (ringing)
- RMS voltage and mean (DC offset)

**Statistical Analysis** (Demo 02):

- Basic statistics (mean, median, std, min, max, range)
- Percentiles and quartiles
- Distribution metrics (skewness, kurtosis, crest factor)
- Histograms and amplitude distributions
- Outlier detection using statistical thresholds

**Spectral Analysis** (Demo 03):

- Fast Fourier Transform (FFT)
- Power Spectral Density (PSD)
- Total Harmonic Distortion (THD)
- Signal-to-Noise Ratio (SNR)
- Signal-to-Noise and Distortion (SINAD)
- Effective Number of Bits (ENOB)
- Spurious-Free Dynamic Range (SFDR)

**Filtering** (Demo 04):

- Low-pass filters (remove high frequencies)
- High-pass filters (remove DC and low frequencies)
- Band-pass filters (select frequency range)
- Band-stop/notch filters (reject specific frequencies)
- Filter types: Butterworth, Chebyshev, Bessel, Elliptic
- Custom filter design

**Triggering** (Demo 05):

- Rising and falling edge detection
- Edge triggers with threshold and hysteresis
- Pulse width triggers
- Glitch detection (narrow pulses)
- Runt pulse detection (incomplete transitions)
- Trigger segment extraction

**Math Operations** (Demo 06):

- Arithmetic (add, subtract, multiply, divide)
- Differentiation (time derivative)
- Integration (time integral)
- FFT (frequency transform)
- Correlation (cross-correlation)
- Peak detection and envelope extraction

---

## Running Demonstrations

### Option 1: Run Individual Demo

Run a single demo to focus on a specific technique:

```bash
# From the project root
python demonstrations/02_basic_analysis/01_waveform_measurements.py

# Or from the demo directory
cd demonstrations/02_basic_analysis
python 01_waveform_measurements.py
```

Expected output: Measurements with IEEE standard compliance validation.

### Option 2: Run All Basic Analysis Demos

Run all six demos in sequence:

```bash
# From the project root
for demo in demonstrations/02_basic_analysis/*.py; do
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

**IEEE 1241-2010** (Analog-to-Digital Converter Testing):

- ENOB (Effective Number of Bits)
- SNR, SINAD measurement methods
- THD calculation and interpretation
- SFDR definition and usage

**IEEE 181-2011** (Waveform and Vector Measurements):

- Rise time and fall time definitions (10-90%)
- Overshoot and undershoot measurement
- Pulse width and duty cycle
- Transitional waveform terminology

**IEEE 1459-2010** (Power Quality):

- Referenced in power analysis workflows
- Harmonic distortion measurement
- RMS calculations

### Time Domain Techniques

**Waveform Characterization**:

- Peak detection algorithms
- Period and frequency estimation
- Transition time measurement
- Pulse parameter extraction

**Statistical Methods**:

- Distribution analysis
- Outlier detection (3-sigma, IQR)
- Percentile-based characterization
- Variability assessment

### Frequency Domain Techniques

**Spectral Analysis**:

- FFT windowing and scaling
- Power spectral density estimation
- Harmonic identification
- Noise floor characterization

**Quality Metrics**:

- SNR calculation from spectrum
- THD from harmonic amplitudes
- SFDR from spurious peaks
- ENOB from SINAD

### Signal Processing

**Filtering**:

- IIR filter design (Butterworth, Chebyshev)
- FIR filter design
- Filter order selection
- Phase response considerations

**Event Detection**:

- Edge detection with hysteresis
- Pulse width qualification
- Glitch and runt detection
- Trigger holdoff and re-arm

**Mathematical Operations**:

- Trace arithmetic
- Differentiation for rate of change
- Integration for cumulative values
- Cross-correlation for delay estimation

---

## Common Issues and Solutions

### "Frequency detection failed"

**Solution**: The signal may not be periodic or have insufficient cycles:

1. Ensure signal has at least 2 complete cycles
2. Check for DC offset obscuring zero crossings
3. Verify sample rate is adequate (10x signal frequency minimum)
4. Use `period()` instead of `frequency()` for noisy signals

### "FFT results show unexpected peaks"

**Solution**: Windowing and sampling issues are common:

1. Apply appropriate window function (Hann, Hamming)
2. Check for spectral leakage (non-integer cycles in capture)
3. Verify sample rate satisfies Nyquist (2x max frequency)
4. Remove DC offset before FFT

### "Filter introduces unexpected artifacts"

**Solution**: Filter design requires careful parameter selection:

1. Check filter order isn't too high (ringing)
2. Verify cutoff frequency is appropriate for signal
3. Consider phase distortion (use Bessel for linear phase)
4. Apply forward-backward filtering (filtfilt) for zero phase

### "Trigger detects false events"

**Solution**: Adjust trigger parameters for noise immunity:

1. Increase hysteresis to reject noise
2. Use holdoff time to prevent re-triggering
3. Apply filtering before triggering
4. Combine multiple trigger conditions (AND/OR logic)

### "Measurements vary between runs"

**Solution**: Noise and measurement window affect results:

1. Check signal-to-noise ratio (SNR)
2. Use longer measurement windows for averaging
3. Apply appropriate filtering
4. Validate with known reference signals

---

## Next Steps: Where to Go After Basic Analysis

### If You Want to...

| Goal                                   | Next Demo                                         | Path                           |
| -------------------------------------- | ------------------------------------------------- | ------------------------------ |
| Decode protocols from analyzed signals | `03_protocol_decoding/01_serial_comprehensive.py` | Analysis → Protocol decoding   |
| Perform advanced jitter analysis       | `04_advanced_analysis/01_jitter_analysis.py`      | Basic → Advanced timing        |
| Analyze power supply quality           | `04_advanced_analysis/02_power_analysis.py`       | Basic → Power quality          |
| Assess signal integrity                | `04_advanced_analysis/03_signal_integrity.py`     | Basic → SI analysis            |
| Generate eye diagrams                  | `04_advanced_analysis/04_eye_diagrams.py`         | Basic → Eye diagrams           |
| Discover unknown patterns              | `04_advanced_analysis/05_pattern_discovery.py`    | Analysis → Pattern recognition |

### Recommended Learning Sequence

1. **Master Basic Analysis** (this section)
   - Understand time and frequency domain
   - Learn measurement techniques
   - Apply filtering and triggering

2. **Explore Protocol Decoding** (03_protocol_decoding/)
   - Apply analysis to extract protocols
   - Use triggering for packet synchronization
   - Validate protocol timing with measurements

3. **Advanced Analysis** (04_advanced_analysis/)
   - Deep-dive into jitter and timing
   - Power quality assessment
   - Signal integrity for high-speed links

4. **Domain-Specific Applications** (05_domain_specific/)
   - Apply analysis to real-world problems
   - Industry-standard compliance
   - Specialized workflows

---

## Tips for Learning

### Understand Before Measuring

Each measurement has assumptions:

```python
# Frequency requires periodic signal
freq = frequency(trace)  # Assumes stable period

# Rise time requires clean edges
rt = rise_time(trace)  # Assumes 10-90% transition visible

# FFT requires sufficient samples
spectrum = fft(trace)  # Assumes power-of-2 length optimal
```

Read the docstrings to understand requirements.

### Visualize Results

Plotting helps understand measurements:

```python
import matplotlib.pyplot as plt

# Plot time domain
plt.plot(trace.time(), trace.data)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")

# Plot frequency domain
freq, mag = fft(trace)
plt.plot(freq, mag)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
```

### Validate with Known Signals

Test measurements on synthetic signals first:

```python
from demonstrations.common import generate_sine_wave

# Known 1 kHz, 2 Vpp sine wave
trace = generate_sine_wave(frequency=1000.0, amplitude=1.0)

# Validate measurements
assert abs(frequency(trace) - 1000.0) < 1.0  # Within 1 Hz
assert abs(amplitude(trace) - 2.0) < 0.01    # Within 10 mV
```

### Combine Techniques

Real analysis combines multiple techniques:

```python
# 1. Filter noise
filtered = low_pass(trace, cutoff=10000.0)

# 2. Detect edges
edges = find_rising_edges(filtered, threshold=0.5)

# 3. Measure timing
periods = [edges[i+1] - edges[i] for i in range(len(edges)-1)]

# 4. Statistical characterization
stats = basic_stats(periods)
print(f"Period jitter: {stats['std']} s")
```

### Reference IEEE Standards

Oscura follows IEEE standards for measurement definitions:

- **IEEE 1241-2010** - ADC testing (SNR, ENOB, THD)
- **IEEE 181-2011** - Waveform measurements (rise time, overshoot)
- Read the standards for authoritative definitions

---

## Understanding the Framework

### Measurement API

**Simple Measurements**:

```python
from oscura import amplitude, frequency, rms

amp = amplitude(trace)      # Peak-to-peak voltage
freq = frequency(trace)     # Fundamental frequency
rms_v = rms(trace)         # RMS voltage
```

**Statistical Measurements**:

```python
from oscura import basic_stats, percentiles

stats = basic_stats(trace)
# Returns: {'mean', 'median', 'std', 'min', 'max', 'range'}

pct = percentiles(trace, [25, 50, 75])
# Returns quartiles
```

**Spectral Measurements**:

```python
from oscura import fft, psd, thd, snr

freq_bins, magnitude = fft(trace)
freq_bins, power = psd(trace)
total_hd = thd(trace, fundamental_freq=1000.0)
signal_noise_ratio = snr(trace, signal_freq=1000.0)
```

### Filtering API

**Pre-designed Filters**:

```python
from oscura import low_pass, high_pass, band_pass, band_stop

# Remove high frequencies
filtered = low_pass(trace, cutoff=10000.0, order=4)

# Remove DC and low frequencies
ac_trace = high_pass(trace, cutoff=100.0)

# Select frequency band
band = band_pass(trace, low=1000.0, high=5000.0)
```

**Custom Filter Design**:

```python
from oscura.filtering.design import design_filter

# Design custom filter
sos = design_filter(
    filter_type='lowpass',
    cutoff=10000.0,
    sample_rate=trace.metadata.sample_rate,
    order=6,
    ftype='butter'  # butterworth, cheby1, cheby2, bessel, ellip
)
```

### Triggering API

**Edge Triggers**:

```python
from oscura import EdgeTrigger, find_rising_edges, find_falling_edges

trigger = EdgeTrigger(threshold=0.5, edge='rising', hysteresis=0.1)
edges = find_rising_edges(trace, threshold=0.5)
```

**Pulse Triggers**:

```python
from oscura import PulseWidthTrigger, find_pulses, find_glitches

pulses = find_pulses(trace, min_width=1e-6, max_width=10e-6)
glitches = find_glitches(trace, max_width=100e-9)
```

---

## Resources

### In This Repository

- **`src/oscura/analyzers/waveform/`** - Measurement implementations
- **`src/oscura/analyzers/statistical/`** - Statistical analysis
- **`src/oscura/filtering/`** - Filter design and application
- **`tests/unit/analyzers/`** - Measurement test cases

### External Resources

- **[IEEE 1241-2010](https://standards.ieee.org/)** - ADC testing standard
- **[IEEE 181-2011](https://standards.ieee.org/)** - Waveform measurement standard
- **[SciPy Signal Processing](https://docs.scipy.org/doc/scipy/reference/signal.html)** - Underlying algorithms
- **[NumPy FFT](https://numpy.org/doc/stable/reference/routines.fft.html)** - FFT documentation

### Getting Help

1. Check demo docstrings for detailed examples
2. Review IEEE standards for measurement definitions
3. Examine source code in `src/oscura/analyzers/`
4. Test with synthetic signals from `demonstrations/common.py`
5. Validate against known reference measurements

---

## Summary

The Basic Analysis section covers:

| Demo                     | Focus                    | Outcome                                           |
| ------------------------ | ------------------------ | ------------------------------------------------- |
| 01_waveform_measurements | Time domain measurements | Amplitude, frequency, rise/fall, duty cycle       |
| 02_statistics            | Distribution analysis    | Mean, std, percentiles, outliers                  |
| 03_spectral_analysis     | Frequency domain         | FFT, PSD, THD, SNR, ENOB, SFDR                    |
| 04_filtering             | Signal conditioning      | Low/high/band-pass, filter design                 |
| 05_triggering            | Event detection          | Edges, pulses, glitches, runts                    |
| 06_math_operations       | Signal manipulation      | Arithmetic, differentiate, integrate, correlation |

After completing these six 80-minute demonstrations, you'll understand:

- How to measure signals in time and frequency domains
- Statistical characterization of signal distributions
- IEEE-compliant measurement techniques
- Filtering for noise reduction and signal conditioning
- Event detection and triggering strategies
- Mathematical operations on waveforms

**Ready to start?** Run this to begin with waveform measurements:

```bash
python demonstrations/02_basic_analysis/01_waveform_measurements.py
```

Happy analyzing!
