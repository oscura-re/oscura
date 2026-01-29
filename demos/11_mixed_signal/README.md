# Mixed-Signal Analysis Demos

Demonstrates **comprehensive mixed-signal analysis** including eye diagrams, jitter characterization, and signal integrity validation per **IEEE 2414-2020** using Oscura's high-speed digital analysis capabilities.

---

## Files in This Demo

1. **`comprehensive_mixed_signal_demo.py`** ⭐ **COMPREHENSIVE**
   - Eye diagram generation with density plots
   - TIE (Time Interval Error) analysis
   - RMS and peak-to-peak jitter measurements
   - Clock recovery (FFT and edge-based)
   - Signal integrity metrics
   - IEEE 2414-2020 compliance validation

---

## Quick Start

### 1. High-Speed Serial Data Analysis

```bash
# Using synthetic serial data
python demos/07_mixed_signal/comprehensive_mixed_signal_demo.py --type serial

# Analyze captured data
python demos/07_mixed_signal/comprehensive_mixed_signal_demo.py \
    --file serial_1gbps.wfm \
    --type serial \
    --all-analysis
```

### 2. Clock Jitter Analysis

```bash
# Analyze clock jitter (IEEE 2414-2020)
python demos/07_mixed_signal/comprehensive_mixed_signal_demo.py \
    --file clock_signal.wfm \
    --type clock \
    --verbose
```

### 3. Eye Diagram Generation

```bash
# Generate eye diagram with measurements
python demos/07_mixed_signal/comprehensive_mixed_signal_demo.py \
    --file prbs_data.wfm \
    --type eye \
    --bit-rate 1e9 \
    --save-plots
```

---

## Analysis Capabilities

### ✅ Eye Diagram Analysis

**Features**:

- Density plot visualization
- Automatic clock recovery
- Eye opening measurements:
  - Eye height (vertical opening)
  - Eye width (horizontal opening)
  - Crossing voltage
  - BER margin estimation
- Configurable bit rate

**Applications**:

- High-speed serial link characterization
- BER prediction
- Signal integrity validation
- Receiver margin analysis

### ✅ Jitter Analysis (IEEE 2414-2020)

**RMS Jitter**:

- Root-mean-square timing variation
- Statistical uncertainty (1-sigma)
- Quality grading: <1 ps (Excellent), <5 ps (Good), <10 ps (Fair)

**Peak-to-Peak Jitter**:

- Maximum timing deviation range
- Worst-case timing margin

**Time Interval Error (TIE)**:

- Edge-by-edge timing deviation
- Time-series jitter data
- Trend analysis capability

**Applications**:

- Clock quality assessment
- PLL performance characterization
- Jitter budgeting for system design

### ✅ Clock Recovery

**FFT Method**:

- Frequency-domain peak detection
- Confidence metric based on spectrum
- Suitable for periodic signals

**Edge Timing Method**:

- Time-domain edge analysis
- Includes jitter statistics
- More accurate for clean clocks

**Applications**:

- Automatic bit rate detection
- Clock quality monitoring
- Data eye diagram synchronization

### ✅ Signal Integrity Metrics

**Measurements**:

- Logic level voltages (HIGH/LOW)
- Signal swing (peak-to-peak)
- Noise levels (σ for each logic level)
- Signal-to-noise ratio (SNR)
- RMS voltage

**Applications**:

- Voltage margin analysis
- Noise characterization
- Link quality assessment

---

## IEEE 2414-2020 Compliance

The demo validates compliance with **IEEE Standard 2414-2020** for jitter and phase noise measurements:

### Validation Checks

1. **RMS Jitter Limits**: Typical spec <10 ps for Gbps links
2. **Measurement Uncertainty**: Should be <10% of measured value
3. **Sample Count**: Recommend ≥100 edges for statistical validity
4. **Method Compliance**: Uses standardized TIE measurement

### Compliance Report

Generates automatic PASS/FAIL report with:

- Jitter specification comparison
- Measurement quality assessment
- Statistical confidence metrics
- Violation summary

---

## Python API Usage

```python
import oscura as osc
from oscura.analyzers.digital.timing import (
    recover_clock_fft,
    recover_clock_edge,
    rms_jitter,
    peak_to_peak_jitter,
    time_interval_error,
)
from oscura.visualization.eye import plot_eye

# Load signal
signal = osc.load("serial_data.wfm")

# Clock recovery
clock_result = recover_clock_fft(signal)
print(f"Clock: {clock_result.frequency / 1e6:.3f} MHz")
print(f"Confidence: {clock_result.confidence:.2f}")

# Edge-based recovery with jitter
edge_result = recover_clock_edge(signal, edge_type="rising")
print(f"RMS jitter: {edge_result.jitter_rms * 1e12:.2f} ps")
print(f"Pk-Pk jitter: {edge_result.jitter_pp * 1e12:.2f} ps")

# Detailed jitter analysis
rms_result = rms_jitter(signal, edge_type="rising")
print(f"RMS jitter: {rms_result.rms * 1e12:.3f} ps")
print(f"Uncertainty: ±{rms_result.uncertainty * 1e12:.3f} ps")
print(f"Samples: {rms_result.samples} edges")

pp_jitter = peak_to_peak_jitter(signal, edge_type="rising")
print(f"Pk-Pk jitter: {pp_jitter * 1e12:.3f} ps")

# Time Interval Error
tie = time_interval_error(signal, edge_type="rising")
print(f"TIE RMS: {np.std(tie) * 1e12:.3f} ps")
print(f"TIE Pk-Pk: {(np.max(tie) - np.min(tie)) * 1e12:.3f} ps")

# Eye diagram
fig = plot_eye(
    signal,
    bit_rate=1e9,  # 1 Gbps
    show_measurements=True,
    save_path="eye.png",
)
```

---

## Use Cases

### High-Speed Serial Links

- PCIe, USB, SATA, DisplayPort
- Ethernet (1G/10G/25G/100G)
- HDMI, MIPI
- SerDes characterization

### Clock Quality Analysis

- PLL jitter measurement
- Oscillator stability
- Clock distribution networks
- Timing margin validation

### Signal Integrity

- Eye diagram validation
- BER prediction
- Receiver sensitivity testing
- Link budget analysis

### Debug and Troubleshooting

- Jitter source identification
- Crosstalk analysis
- Power supply noise effects
- PCB trace quality

---

## Output Interpretation

### RMS Jitter Quality

- **<1 ps**: Excellent (precision timing, high-speed links)
- **<5 ps**: Good (Gbps serial links)
- **<10 ps**: Fair (100 Mbps - 1 Gbps links)
- **>10 ps**: Poor (investigate jitter sources)

### Eye Diagram Quality

- **Eye Height >70% of swing**: Excellent margin
- **Eye Width >0.5 UI**: Good timing margin
- **BER Margin >20%**: Low error probability
- **Clean crossing**: Low deterministic jitter

### Signal-to-Noise Ratio

- **>30 dB**: Excellent (low noise)
- **>20 dB**: Good (adequate margin)
- **>15 dB**: Fair (marginal)
- **<15 dB**: Poor (high BER risk)

---

## Common Issues

### Issue: "Clock recovery failed"

**Solution**: Check signal levels, ensure edges are present, reduce noise

### Issue: "High RMS jitter detected"

**Solution**: Investigate power supply noise, crosstalk, PCB quality

### Issue: "Closed eye diagram"

**Solution**: Check ISI (inter-symbol interference), jitter, noise levels

### Issue: "Insufficient edges for TIE"

**Solution**: Increase capture duration, ensure clock is present

---

## Advanced Topics

### Jitter Decomposition (RJ/DJ)

While not automatically separated in this demo, you can analyze:

- **RJ (Random Jitter)**: Gaussian component (from noise)
- **DJ (Deterministic Jitter)**: Repeatable component (from ISI, crosstalk)
- Use TIE histogram analysis to identify components

### S-Parameters (Conceptual)

Oscura focuses on time-domain analysis. For S-parameters:

- Use TDR (Time Domain Reflectometry) from step response
- Convert to frequency domain via FFT
- Analyze impedance discontinuities

### TDR Analysis

Time Domain Reflectometry can be performed by:

1. Generate step input
2. Measure reflections
3. Calculate impedance: Z = Z₀(1+ρ)/(1-ρ)
4. Identify discontinuities

---

## Performance Notes

- Eye diagram generation is compute-intensive for long captures
- Use density plots (default) for better visualization
- TIE analysis requires sufficient edges (≥100 recommended)
- FFT clock recovery is faster but less accurate than edge method

---

## Related Documentation

- **Main demos**: `demos/README.md`
- **Spectral compliance**: `demos/06_spectral_compliance/`
- **Examples**: `examples/02_digital_analysis/`
- **IEEE 2414-2020**: Standard for jitter and phase noise measurements

---

## References

### IEEE Standards

- **IEEE 2414-2020**: Standard for Jitter and Phase Noise
- **IEEE 181-2011**: Standard for Transitional Waveform Definitions
- **JEDEC Standard No. 65B**: High-Speed Interface Eye Diagram Measurements

### Application Notes

- **TDR/TDT Theory**: Time Domain Reflectometry/Transmission
- **Eye Diagram Fundamentals**: Signal integrity analysis
- **Jitter Analysis**: RJ/DJ decomposition methods

---

**Last Updated**: 2026-01-15
**Status**: Production-ready
