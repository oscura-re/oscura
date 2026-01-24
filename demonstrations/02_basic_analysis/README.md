# Basic Analysis

**Master fundamental signal measurements and analysis techniques for hardware reverse engineering.**

This section contains 6 demonstrations covering waveform measurements, statistical analysis, spectral analysis, filtering, triggering, and mathematical operations. Learn the essential analysis techniques used in every reverse engineering workflow.

---

## Prerequisites

See [main demonstrations README](../README.md#installation) for installation instructions.

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

## Running the Demonstrations

See [main demonstrations README](../README.md#running-demonstrations) for all execution options.

**Category-specific tip:** Start with the first demonstration (e.g., `01_waveform_measurements.py`) before exploring advanced examples.

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

- **Understand assumptions**: Frequency needs periodic signals, rise time needs clean edges, FFT needs sufficient samples
- **Validate with known signals**: Test measurements first on synthetic signals with known properties
- **Combine techniques**: Real analysis uses filtering, edge detection, timing measurement, and statistics together
- **Follow IEEE standards**: IEEE 1241-2010 (ADC), IEEE 181-2011 (waveforms) define authoritative measurement methods

## Core APIs

```python
from oscura import amplitude, frequency, rms, basic_stats, percentiles
from oscura import fft, psd, thd, snr
from oscura import low_pass, high_pass, band_pass, band_stop
from oscura import find_rising_edges, find_falling_edges, find_pulses, find_glitches

amp = amplitude(trace)  # Peak-to-peak
freq = frequency(trace) # Fundamental frequency
rms_v = rms(trace)     # RMS voltage
filtered = low_pass(trace, cutoff=10000.0)  # Low-pass filter
edges = find_rising_edges(trace, 0.5)       # Edge detection
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
