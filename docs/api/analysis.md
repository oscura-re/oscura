# Analysis API Reference

> **Version**: 0.1.0
> **Last Updated**: 2026-01-08

## Overview

Oscura provides comprehensive signal analysis capabilities including waveform measurements, spectral analysis, digital timing analysis, protocol decoding, and statistical analysis.

## Quick Start

```python
import oscura as osc

# Load and analyze
trace = osc.load("capture.wfm")

# Basic measurements
freq = osc.frequency(trace)
rise = osc.rise_time(trace)
amp = osc.amplitude(trace)

# Signal classification
classification = osc.classify_signal(trace)
print(f"Signal type: {classification['signal_type']}")

# Full analysis report
from oscura.reporting import analyze
results = analyze(trace)
```

## Waveform Measurements

### Timing Measurements

```python
import oscura as osc

trace = osc.load("signal.wfm")

# Frequency and period
freq = osc.frequency(trace)           # Hz
period = osc.period(trace)            # seconds

# Edge timing (10%-90% by default)
rise = osc.rise_time(trace)           # seconds
fall = osc.fall_time(trace)           # seconds

# Custom thresholds
rise = osc.rise_time(trace, low_pct=20, high_pct=80)

# Duty cycle
duty = osc.duty_cycle(trace)          # 0.0-1.0
pulse_width = osc.pulse_width(trace)  # seconds
```

### Amplitude Measurements

```python
# Peak-to-peak amplitude
amp = osc.amplitude(trace)            # volts

# Overshoot/undershoot (percentage)
overshoot = osc.overshoot(trace)      # %
undershoot = osc.undershoot(trace)    # %
preshoot = osc.preshoot(trace)        # %

# DC levels
mean_val = osc.mean(trace)            # volts
rms_val = osc.rms(trace)              # volts
```

### Statistical Measurements

```python
# Basic statistics
stats = osc.statistics(trace)
print(f"Mean: {stats['mean']}")
print(f"Std Dev: {stats['std']}")
print(f"Min: {stats['min']}")
print(f"Max: {stats['max']}")
```

## Signal Intelligence

Oscura provides intelligent signal classification and measurement suitability checking.

### `classify_signal()`

Classify signal type and characteristics.

```python
import oscura as osc

trace = osc.load("signal.wfm")
info = osc.classify_signal(trace)

print(f"Type: {info['signal_type']}")           # "digital", "analog", "mixed", "dc"
print(f"Is periodic: {info['is_periodic']}")    # True/False
print(f"Characteristics: {info['characteristics']}")  # ["periodic", "clean", ...]
print(f"Frequency: {info['frequency_estimate']}")     # Hz or None
print(f"SNR: {info['snr_db']}")                       # dB or None
print(f"Confidence: {info['confidence']}")            # 0.0-1.0
```

**Parameters:**

|Parameter|Type|Default|Description|
|---|---|---|---|
|`trace`|`WaveformTrace \|ndarray`|required|Input waveform|
|`sample_rate`|`float`|`1.0`|Sample rate (if ndarray)|
|`digital_threshold_ratio`|`float`|`0.8`|Threshold for digital detection|
|`dc_threshold_percent`|`float`|`90.0`|DC component threshold|
|`periodicity_threshold`|`float`|`0.7`|Periodicity correlation threshold|

### `assess_signal_quality()`

Assess signal quality metrics.

```python
import oscura as osc

trace = osc.load("signal.wfm")
quality = osc.assess_signal_quality(trace)

print(f"SNR: {quality['snr']} dB")
print(f"Noise level: {quality['noise_level']}")
print(f"Clipping: {quality['clipping']}")
print(f"Saturation: {quality['saturation']}")
print(f"Warnings: {quality['warnings']}")
print(f"Dynamic range: {quality['dynamic_range']} dB")
print(f"Crest factor: {quality['crest_factor']}")
```

### `check_measurement_suitability()`

Check if a measurement is suitable for a signal.

```python
import oscura as osc

trace = osc.load("dc_signal.wfm")
check = osc.check_measurement_suitability(trace, "frequency")

print(f"Suitable: {check['suitable']}")
print(f"Confidence: {check['confidence']}")
print(f"Expected result: {check['expected_result']}")  # "valid", "nan", "unreliable"
print(f"Warnings: {check['warnings']}")
print(f"Suggestions: {check['suggestions']}")
```

### `suggest_measurements()`

Get recommended measurements for a signal.

```python
import oscura as osc

trace = osc.load("square_wave.wfm")
suggestions = osc.suggest_measurements(trace)

for s in suggestions[:5]:
    print(f"{s['name']}: {s['rationale']}")
    print(f"  Category: {s['category']}, Priority: {s['priority']}")
    print(f"  Confidence: {s['confidence']}")
```

## Spectral Analysis

### FFT Analysis

```python
import oscura as osc

trace = osc.load("signal.wfm")

# Compute FFT
fft_result = osc.fft(trace)
freqs = fft_result['frequencies']
magnitudes = fft_result['magnitudes']

# Power spectral density
psd = osc.psd(trace)

# Spectrogram (time-frequency)
spectrogram = osc.spectrogram(trace)
```

### Spectral Measurements

```python
# Total Harmonic Distortion
thd = osc.thd(trace)                  # %

# Signal-to-Noise Ratio
snr = osc.snr(trace)                  # dB

# SINAD (Signal to Noise and Distortion)
sinad = osc.sinad(trace)              # dB

# Effective Number of Bits
enob = osc.enob(trace)                # bits

# Spurious-Free Dynamic Range
sfdr = osc.sfdr(trace)                # dB
```

## Digital Analysis

### Timing Analysis

```python
from oscura.analyzers.digital import timing

# Setup/hold time analysis
setup = timing.setup_time(data, clock, data_signal)
hold = timing.hold_time(data, clock, data_signal)

# Skew measurement
skew = timing.skew(signal_a, signal_b)

# Clock recovery
clock_info = timing.recover_clock(trace)
print(f"Clock frequency: {clock_info['frequency']} Hz")
print(f"Jitter: {clock_info['jitter_rms']} seconds")
```

### Edge Analysis

```python
from oscura.analyzers.digital import edges

# Find edges
rising_edges = edges.find_rising_edges(trace)
falling_edges = edges.find_falling_edges(trace)

# Edge rate
slew_rate = edges.slew_rate(trace)
```

<a id="protocol-decoding"></a>

## Protocol Decoders

Oscura includes protocol decoders for common serial protocols.

### UART

```python
from oscura.analyzers.protocols import uart

decoded = uart.decode(trace, baud_rate=115200)
for frame in decoded.frames:
    print(f"Data: {frame.data}, Time: {frame.timestamp}")
```

### SPI

```python
from oscura.analyzers.protocols import spi

decoded = spi.decode(
    clk=channels['clk'],
    mosi=channels['mosi'],
    miso=channels['miso'],
    cs=channels['cs']
)
```

### I2C

```python
from oscura.analyzers.protocols import i2c

decoded = i2c.decode(
    sda=channels['sda'],
    scl=channels['scl']
)
for transaction in decoded.transactions:
    print(f"Address: {transaction.address}, Data: {transaction.data}")
```

### CAN

```python
from oscura.analyzers.protocols import can

decoded = can.decode(trace, bit_rate=500000)
for frame in decoded.frames:
    print(f"ID: {frame.id:03X}, Data: {frame.data.hex()}")
```

### Additional Protocols

- `can_fd` - CAN FD
- `lin` - LIN bus
- `flexray` - FlexRay
- `i2s` - I2S audio
- `jtag` - JTAG
- `swd` - Serial Wire Debug
- `usb` - USB
- `manchester` - Manchester encoding
- `onewire` - 1-Wire
- `hdlc` - HDLC framing

<a id="protocol-inference"></a>

## Pattern Analysis

### Pattern Discovery

```python
from oscura.analyzers.patterns import discovery

patterns = discovery.find_patterns(trace)
for p in patterns:
    print(f"Pattern at {p.start}, length {p.length}, repeats {p.count}")
```

### Sequence Analysis

```python
from oscura.analyzers.patterns import sequences

# Find repeating sequences
repeats = sequences.find_repeating(data)

# Detect periodic patterns
periodic = sequences.detect_periodic(data)
```

## Statistical Analysis

### Correlation

```python
from oscura.analyzers.statistics import correlation

# Cross-correlation
xcorr = correlation.cross_correlate(signal_a, signal_b)

# Auto-correlation
acorr = correlation.auto_correlate(signal)
```

### Distribution Analysis

```python
from oscura.analyzers.statistics import distribution

# Fit distribution
fit = distribution.fit(data)
print(f"Best fit: {fit['distribution']}")
print(f"Parameters: {fit['parameters']}")

# Histogram
hist = distribution.histogram(data, bins=100)
```

### Outlier Detection

```python
from oscura.analyzers.statistics import outliers

outlier_indices = outliers.detect(data, method='zscore', threshold=3.0)
```

### Entropy Analysis

```python
from oscura.analyzers.statistical import entropy

# Shannon entropy
h = entropy.shannon_entropy(data)

# Sliding window entropy
sliding = entropy.sliding_entropy(data, window_size=256)
```

## Jitter Analysis

```python
from oscura.analyzers.jitter import measurements

# RMS jitter
jitter_rms = measurements.rms_jitter(trace)

# Peak-to-peak jitter
jitter_pp = measurements.peak_to_peak_jitter(trace)

# Jitter decomposition
decomp = measurements.decompose_jitter(trace)
print(f"Random jitter: {decomp['random']}")
print(f"Deterministic jitter: {decomp['deterministic']}")
```

## Eye Diagram Analysis

```python
from oscura.analyzers.eye import diagram, metrics

# Generate eye diagram
eye = diagram.create_eye_diagram(trace, bit_rate=1e9)

# Eye metrics
eye_metrics = metrics.measure_eye(eye)
print(f"Eye height: {eye_metrics['height']}")
print(f"Eye width: {eye_metrics['width']}")
print(f"Eye opening: {eye_metrics['opening']}")
```

## Power Analysis

```python
from oscura.analyzers.power import basic, ripple

# Power measurements
power = basic.calculate_power(voltage, current)

# Ripple analysis
ripple_result = ripple.analyze_ripple(trace)
print(f"Ripple Vpp: {ripple_result['vpp']}")
print(f"Ripple frequency: {ripple_result['frequency']}")
```

## Comprehensive Analysis

### Using the Analysis Engine

```python
from oscura.reporting import analyze, AnalysisDomain

# Run specific analyses
results = analyze(
    trace,
    domains=[
        AnalysisDomain.WAVEFORM,
        AnalysisDomain.SPECTRAL,
        AnalysisDomain.STATISTICS
    ]
)

# Access results
print(f"Frequency: {results['waveform']['frequency']}")
print(f"THD: {results['spectral']['thd']}")
```

### Smart Analysis Selection

```python
from oscura.inference.signal_intelligence import recommend_analyses

recommendations = recommend_analyses(trace.data, trace.metadata.sample_rate)
for rec in recommendations:
    print(f"{rec.domain.value}: {rec.reasoning}")
    print(f"  Priority: {rec.priority}, Confidence: {rec.confidence}")
```

## NaN Handling

Many measurements return `NaN` when not applicable. See:


```python
import numpy as np
import oscura as osc

result = osc.frequency(trace)
if np.isnan(result):
    # Check why
    suitability = osc.check_measurement_suitability(trace, "frequency")
    print(f"Not suitable: {suitability['warnings']}")
```

## See Also

- [Loader API](loader.md) - Data loading
- [Component Analysis API](component-analysis.md) - TDR, capacitance, inductance measurements
- [Export API](export.md) - Data export
- [Reporting API](reporting.md) - Report generation
