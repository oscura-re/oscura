# Category 02: Basic Analysis

Fundamental signal analysis techniques for oscilloscope data.

## Overview

This category contains 8 comprehensive demonstrations covering essential waveform analysis, digital signal analysis, spectral analysis, filtering, triggering, cursor measurements, and statistical characterization. These demos form the foundation for understanding Oscura's analysis capabilities.

## Demonstrations

### 01_waveform_basics.py - Waveform Measurements Basics
**Time:** ~5 minutes | **Level:** Beginner

Core waveform parameter measurements including:
- Amplitude (peak-to-peak, RMS, average)
- Frequency and period measurement
- Rise/fall time analysis
- Duty cycle measurement
- Overshoot/undershoot detection

**IEEE Standards:** IEEE 181-2011 (Pulse measurement terminology)

**Usage:**
```bash
python demos/02_basic_analysis/01_waveform_basics.py
python demos/02_basic_analysis/01_waveform_basics.py --verbose
```

---

### 02_digital_basics.py - Digital Signal Analysis Basics
**Time:** ~5 minutes | **Level:** Beginner

Digital signal analysis fundamentals:
- Edge detection (rising/falling)
- Pulse width measurement
- Setup and hold time analysis
- Propagation delay measurement
- Logic level detection

**IEEE Standards:** IEEE 181-2011

**Usage:**
```bash
python demos/02_basic_analysis/02_digital_basics.py
```

---

### 03_spectral_basics.py - Spectral Analysis Basics
**Time:** ~7 minutes | **Level:** Beginner to Intermediate

FFT and frequency domain analysis:
- FFT computation and visualization
- Power Spectral Density (PSD)
- Total Harmonic Distortion (THD)
- Harmonic analysis and detection
- Frequency peak identification
- Signal-to-Noise Ratio (SNR)

**IEEE Standards:** IEEE 1241-2010, IEEE 1057

**Usage:**
```bash
python demos/02_basic_analysis/03_spectral_basics.py
```

---

### 04_measurements.py - Comprehensive Measurement Suite
**Time:** ~8 minutes | **Level:** Intermediate

Complete IEEE 181-2011 compliant measurement suite:
- All pulse measurements (width, period, frequency)
- Amplitude measurements (peak, RMS, average, min, max)
- Timing measurements (rise/fall time, slew rate)
- Overshoot and undershoot detection
- Statistical waveform characterization
- Measurement report generation

**IEEE Standards:** IEEE 181-2011 (comprehensive coverage)

**Usage:**
```bash
python demos/02_basic_analysis/04_measurements.py
```

---

### 05_filtering.py - Signal Filtering Techniques
**Time:** ~6 minutes | **Level:** Beginner to Intermediate

Comprehensive filtering capabilities:
- Low-pass filtering (remove high-frequency noise)
- High-pass filtering (remove DC offset)
- Band-pass filtering (isolate frequency band)
- Band-stop/notch filtering (remove specific frequencies)
- Moving average filtering
- Practical noise reduction examples

**IEEE Standards:** IEEE 181-2011

**Usage:**
```bash
python demos/02_basic_analysis/05_filtering.py
```

---

### 06_triggers.py - Trigger Detection and Analysis
**Time:** ~6 minutes | **Level:** Intermediate

Oscilloscope-style trigger analysis:
- Edge triggers (rising/falling/both)
- Level triggers (threshold crossing)
- Pulse width triggers (narrow/wide pulses)
- Pattern triggers (specific bit patterns)
- Trigger holdoff and count
- Pre/post-trigger capture concepts

**Usage:**
```bash
python demos/02_basic_analysis/06_triggers.py
```

---

### 07_cursors.py - Cursor Measurements
**Time:** ~5 minutes | **Level:** Beginner

Oscilloscope-style cursor measurements:
- Time cursors (delta-t measurements)
- Voltage cursors (delta-v measurements)
- Cursor-to-cursor calculations
- Reference markers
- Multiple cursor pairs
- Frequency from period cursors

**Usage:**
```bash
python demos/02_basic_analysis/07_cursors.py
```

---

### 08_statistics.py - Statistical Signal Analysis
**Time:** ~7 minutes | **Level:** Intermediate

Statistical characterization of signals:
- Mean, median, standard deviation
- Histogram analysis and binning
- Distribution shape (skewness, kurtosis)
- Correlation analysis
- Signal-to-Noise Ratio (SNR)
- Coefficient of variation
- Percentile analysis

**Usage:**
```bash
python demos/02_basic_analysis/08_statistics.py
```

---

## Quick Start

Run all demos in sequence:
```bash
for demo in demos/02_basic_analysis/0*.py; do
    python "$demo" || break
done
```

Run specific demo with verbose output:
```bash
python demos/02_basic_analysis/01_waveform_basics.py --verbose
```

## Key Capabilities Demonstrated

### Time Domain Analysis
- `oscura.amplitude()` - Peak-to-peak voltage
- `oscura.frequency()` - Frequency measurement
- `oscura.period()` - Period measurement
- `oscura.rise_time()` - Rising edge transition time
- `oscura.fall_time()` - Falling edge transition time
- `oscura.duty_cycle()` - Duty cycle percentage
- `oscura.pulse_width()` - Pulse width measurement
- `oscura.mean()` - DC offset/average value
- `oscura.rms()` - RMS voltage

### Digital Analysis
- `oscura.find_edges()` - Edge detection
- `oscura.edge_trigger()` - Edge-based triggering
- `oscura.level_trigger()` - Level-based triggering
- `oscura.pulse_trigger()` - Pulse width triggering
- `oscura.pattern_trigger()` - Pattern matching

### Frequency Domain Analysis
- `oscura.fft()` - Fast Fourier Transform
- `oscura.psd()` - Power Spectral Density
- `oscura.thd()` - Total Harmonic Distortion
- `oscura.snr()` - Signal-to-Noise Ratio
- `oscura.find_peaks()` - Peak detection

### Filtering
- `oscura.low_pass()` - Low-pass filter
- `oscura.high_pass()` - High-pass filter
- `oscura.band_pass()` - Band-pass filter
- `oscura.band_stop()` - Band-stop/notch filter
- `oscura.moving_average()` - Moving average filter

### Statistical Analysis
- `oscura.median()` - Median value
- `oscura.std()` - Standard deviation
- `oscura.histogram()` - Histogram generation
- `oscura.correlation()` - Correlation coefficient
- `oscura.skewness()` - Distribution skewness
- `oscura.kurtosis()` - Distribution kurtosis
- `oscura.percentile()` - Percentile values

### Cursor Measurements
- `oscura.cursor_time_delta()` - Time difference
- `oscura.cursor_voltage_delta()` - Voltage difference
- Reference marker capabilities

## IEEE Standards Compliance

These demonstrations follow industry standards:
- **IEEE 181-2011:** Pulse measurement and analysis terminology
- **IEEE 1241-2010:** ADC testing and characterization
- **IEEE 1057:** Digitizing waveform recorders

## Related Categories

- **00_getting_started:** Introduction to Oscura basics
- **03_protocol_decoding:** Apply measurements to protocol analysis
- **04_advanced_analysis:** Advanced measurement techniques
- **12_standards_compliance:** IEEE standards validation

## Common Patterns

All demos in this category use the enhanced `BaseDemo` pattern:

```python
from demos.common import BaseDemo, ValidationSuite, run_demo_main

class MyAnalysisDemo(BaseDemo):
    name = "Analysis Demo"
    description = "What this demo does"
    category = "basic_analysis"

    capabilities = ["oscura.function1", "oscura.function2"]
    ieee_standards = ["IEEE 181-2011"]
    related_demos = ["../other/demo.py"]

    def generate_data(self):
        # Generate or load test data
        pass

    def run_analysis(self):
        # Perform analysis
        pass

    def validate_results(self, suite: ValidationSuite):
        # Validate results
        suite.check_range("Measurement", value, min, max)

if __name__ == "__main__":
    sys.exit(run_demo_main(MyAnalysisDemo))
```

## Validation

All demos include comprehensive validation suites to ensure:
- Measurements are within expected ranges
- Signal processing is accurate
- IEEE standards compliance
- Numerical stability

## Troubleshooting

**Import errors:**
```bash
# Ensure you're in the project root
cd /path/to/oscura
python demos/02_basic_analysis/01_waveform_basics.py
```

**Missing dependencies:**
```bash
uv sync --all-extras
```

**Validation failures:**
Use `--verbose` flag to see detailed execution:
```bash
python demos/02_basic_analysis/01_waveform_basics.py --verbose
```

## Performance

| Demo | Execution Time | Memory | Complexity |
|------|---------------|--------|------------|
| 01_waveform_basics.py | ~2s | <50MB | Low |
| 02_digital_basics.py | ~2s | <50MB | Low |
| 03_spectral_basics.py | ~3s | <100MB | Medium |
| 04_measurements.py | ~3s | <50MB | Medium |
| 05_filtering.py | ~3s | <100MB | Medium |
| 06_triggers.py | ~2s | <50MB | Low |
| 07_cursors.py | ~2s | <50MB | Low |
| 08_statistics.py | ~3s | <50MB | Medium |

## Learning Path

1. **Start here:** `01_waveform_basics.py` - Learn fundamental measurements
2. **Digital signals:** `02_digital_basics.py` - Understand edge detection
3. **Frequency domain:** `03_spectral_basics.py` - Learn FFT and spectral analysis
4. **Comprehensive:** `04_measurements.py` - IEEE 181 full suite
5. **Noise reduction:** `05_filtering.py` - Signal cleanup techniques
6. **Event detection:** `06_triggers.py` - Trigger mechanisms
7. **Manual measurements:** `07_cursors.py` - Interactive-style analysis
8. **Statistics:** `08_statistics.py` - Statistical characterization

## Contributing

When adding new basic analysis demos:
1. Follow the `BaseDemo` pattern
2. Include comprehensive validation
3. Reference IEEE standards where applicable
4. Add to this README
5. Update `__init__.py`
6. Cross-reference related demos

## License

Part of the Oscura project. See main LICENSE file.
