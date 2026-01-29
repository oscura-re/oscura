# Demo 09: EMC/EMI Compliance Testing

> **Time**: 15-20 minutes
> **Prerequisites**: Basic understanding of EMC/EMI concepts
> **Standards**: CISPR 32, FCC Part 15, IEC 61000-3-2/3-3, IEEE 1459, MIL-STD-461G

Comprehensive demonstration of electromagnetic compatibility (EMC) and electromagnetic interference (EMI) compliance testing capabilities across multiple international standards.

## Overview

This demo showcases Oscura's comprehensive EMC/EMI testing toolkit for product certification, pre-compliance testing, and troubleshooting. It covers:

1. **Conducted Emissions (CE)** - CISPR 32, MIL-STD-461G
2. **Radiated Emissions (RE)** - FCC Part 15, CISPR 32
3. **Power Quality** - IEC 61000-3-2, IEEE 1459
4. **ESD Immunity** - IEC 61000-4-2
5. **EMI Fingerprinting** - Troubleshooting and source identification

## Standards Covered

### CISPR 32 (Multimedia Equipment Emissions)

**Scope**: Conducted and radiated emissions from multimedia equipment
**Frequency Range**:

- Conducted: 150 kHz - 30 MHz
- Radiated: 30 MHz - 6 GHz

**Classes**:

- **Class A**: Industrial/commercial environments
- **Class B**: Residential environments (stricter limits)

**Measurements**:

- Quasi-Peak (QP) detector
- Average (AV) detector

### FCC Part 15 (Radio Frequency Devices)

**Scope**: Unintentional radiators (digital devices)
**Frequency Range**: 30 MHz - 40 GHz

**Classes**:

- **Class A**: Commercial/industrial use
- **Class B**: Residential use (stricter)

**Test Distance**: 3 meters (Class B), 10 meters (Class A)

### IEC 61000-3-2 (Harmonic Current Limits)

**Scope**: Equipment input current ≤ 16 A per phase
**Frequency Range**: 50/60 Hz harmonics (up to 40th harmonic)

**Equipment Classes**:

- **Class A**: Balanced 3-phase equipment
- **Class B**: Portable tools
- **Class C**: Lighting equipment
- **Class D**: Equipment with special current waveform

### IEC 61000-3-3 (Voltage Flicker)

**Scope**: Equipment ≤ 16 A causing voltage fluctuations
**Measurements**:

- Short-term flicker severity (Pst)
- Long-term flicker severity (Plt)

### IEEE 1459 (Power Quality Measurements)

**Scope**: Definitions for power quality measurement
**Measurements**:

- Active, reactive, apparent power
- Power factor (displacement and distortion)
- Total Harmonic Distortion (THD)
- Unbalance factors

### MIL-STD-461G (Military EMI Requirements)

**Scope**: Military and aerospace equipment
**Test CE102**: Conducted emissions, power leads (10 kHz - 10 MHz)

**Stricter than commercial standards** - designed for harsh electromagnetic environments

## Use Cases

### 1. Product Certification

Pre-compliance testing before formal certification:

```python
# Test conducted emissions
signal = osc.load("power_supply_emissions.wfm")
results = analyze_conducted_emissions(
    signal,
    "cispr32_class_b.csv",
    standard="CISPR 32 Class B"
)

if results["compliant"]:
    print("Ready for formal testing")
else:
    print(f"Fix {results['violations']} violations first")
```

**Benefits**:

- Identify issues early
- Reduce certification costs
- Faster time-to-market

### 2. Design Validation

Verify EMC performance during design:

```python
# Check margins on prototype
compliance = check_compliance(
    frequencies, levels,
    limit_freq, limit_values,
    margin_db=6.0  # Require 6 dB margin
)

print(f"Minimum margin: {compliance['min_margin']:.1f} dB")
```

**Design Goals**:

- 6 dB margin for production variation
- Identify worst-case frequencies
- Optimize filtering/shielding

### 3. Troubleshooting

Identify EMI sources and fix problems:

```python
# EMI fingerprinting
fingerprint = emc_fingerprinting(emission_signal)

# Identifies:
# - Switching frequencies
# - Clock harmonics
# - Resonance peaks
# - Likely sources
```

**Troubleshooting Workflow**:

1. Capture emissions
2. Identify peak frequencies
3. Correlate with switching frequencies
4. Localize source (near-field probing)
5. Implement countermeasures
6. Re-test

### 4. Comparative Analysis

Compare before/after modifications:

```python
# Before filter
before = analyze_conducted_emissions(before_signal, limits)

# After filter
after = analyze_conducted_emissions(after_signal, limits)

# Compare
improvement = after["min_margin"] - before["min_margin"]
print(f"Improvement: {improvement:.1f} dB")
```

## Files in This Demo

### Scripts

- **`comprehensive_emc_demo.py`** - Main demo script (600+ lines)
  - Automated compliance testing
  - Multiple standards support
  - Detailed reporting

- **`generate_demo_data.py`** - Synthetic data generator
  - Realistic EMI signatures
  - Known violations for demonstration
  - Multiple test scenarios

### Data Files (Generated)

- **`conducted_emissions_ac_line.wfm`** - AC line conducted emissions
  - Switching power supply EMI
  - 150 kHz - 30 MHz
  - CISPR 32 test

- **`radiated_emissions_scan.csv`** - Radiated emissions scan
  - 30 MHz - 1 GHz sweep
  - Peak detector data
  - FCC Part 15 test

- **`power_quality_harmonics.wfm`** - AC power harmonics
  - Non-linear load (rectifier)
  - 60 Hz fundamental
  - IEC 61000-3-2 test

- **`esd_transient_burst.wfm`** - ESD event capture
  - 8 kV contact discharge
  - 10 GSa/s capture
  - IEC 61000-4-2 characterization

- **`mil_std_461_ce102.wfm`** - Military conducted emissions
  - 10 kHz - 10 MHz
  - Stricter limits than commercial

### Compliance Limits

Located in `compliance_limits/` subdirectory:

- **`cispr32_class_b.csv`** - CISPR 32 Class B conducted limits
- **`fcc_part15_class_b.csv`** - FCC Part 15 Class B radiated limits
- **`iec61000-3-2.csv`** - IEC 61000-3-2 harmonic current limits
- **`mil_std_461g_ce102.csv`** - MIL-STD-461G CE102 limits

## Running the Demo

### Step 1: Generate Demo Data

```bash
cd examples/demos/09_emc_compliance
uv run python generate_demo_data.py
```

**Output**: Creates all waveform files and compliance limit files

**Time**: ~5 seconds

### Step 2: Run Comprehensive Demo

```bash
uv run python comprehensive_emc_demo.py
```

**Output**: Detailed compliance analysis for all standards

**Time**: ~2-3 minutes

## Expected Output

### Conducted Emissions Analysis

```
======================================================================
Conducted Emissions Analysis - CISPR 32 Class B
======================================================================

Signal characteristics:
  Sample rate: 200 MSa/s
  Duration: 100.0 µs
  Frequency resolution: 10.00 kHz
  Frequency range: 0 kHz - 100.0 MHz

Compliance frequency range: 150 kHz - 30 MHz

Compliance result: FAIL
  Margin (min): -8.3 dB
  Margin (max): 24.7 dB
  Worst-case frequency: 1.700 MHz

Violations found: 4

Top 5 violations:
  1. 1.700 MHz: 64.3 dBµV (limit: 56.0 dBµV, margin: -8.3 dB)
  2. 0.500 MHz: 58.9 dBµV (limit: 56.0 dBµV, margin: -2.9 dB)
  3. 4.200 MHz: 57.8 dBµV (limit: 56.0 dBµV, margin: -1.8 dB)
  4. 8.900 MHz: 56.4 dBµV (limit: 56.0 dBµV, margin: -0.4 dB)

Top 10 emission peaks:
   1.  1.700 MHz:  64.3 dBµV (margin:  -8.3 dB)
   2.  0.500 MHz:  58.9 dBµV (margin:  -2.9 dB)
   3.  4.200 MHz:  57.8 dBµV (margin:  -1.8 dB)
   ...
```

### Power Quality Analysis

```
======================================================================
Power Quality Harmonics Analysis - IEC 61000-3-2 Class A
======================================================================

Detected fundamental: 60.00 Hz
Total Harmonic Distortion (THD): 38.45%

Harmonic content:
  Order  Frequency    Amplitude
     1      60.0 Hz   100.0%
     3     180.0 Hz    30.0%
     5     300.0 Hz    20.0%
     7     420.0 Hz    12.0%
     9     540.0 Hz     8.0%
    11     660.0 Hz     6.0%
    13     780.0 Hz     4.0%

IEC 61000-3-2 Class A compliance check:
  H 3:  3.00 A /  2.30 A - FAIL
  H 5:  2.00 A /  1.14 A - FAIL
  H 7:  1.20 A /  0.77 A - FAIL
  H 9:  0.80 A /  0.40 A - FAIL

Overall compliance: FAIL
Violations: 4
```

### EMI Fingerprinting

```
======================================================================
EMI Fingerprinting for Troubleshooting
======================================================================

Detected 47 emission peaks

Inferred switching frequency: 100.0 kHz
  (Based on median harmonic spacing)

Identified harmonics of 100.0 kHz:
  H  1:   0.100 MHz,   52.3 dB
  H  2:   0.200 MHz,   46.8 dB
  H  3:   0.300 MHz,   43.1 dB
  H  5:   0.500 MHz,   58.9 dB  <-- Resonance peak
  H  8:   0.800 MHz,   38.2 dB
  H 17:   1.700 MHz,   64.3 dB  <-- Major resonance
  ...

Likely EMI sources:
  - DC-DC converter (50-200 kHz): 58.9 dB
  - High-frequency DC-DC (300 kHz - 1 MHz): 52.4 dB
  - Clock harmonics / digital noise: 48.7 dB
```

### Compliance Summary Report

```
======================================================================
EMC COMPLIANCE SUMMARY REPORT
======================================================================

--- Test Results Overview ---
  CISPR 32 Class B                        : FAIL
    - Violations: 4
    - Minimum margin: -8.3 dB
  MIL-STD-461G CE102                      : FAIL
    - Violations: 3
    - Minimum margin: -5.2 dB
  FCC Part 15 Class B                     : FAIL
    - Violations: 2
    - Minimum margin: -4.1 dB
  IEC 61000-3-2 Class A                   : FAIL
    - Violations: 4
    - Minimum margin: N/A
  IEC 61000-4-2 ESD                       : PASS

======================================================================
OVERALL COMPLIANCE: FAIL
======================================================================

--- Required Actions ---
1. Review violation frequencies and identify sources
2. Implement filtering/shielding at identified frequencies
3. Re-test after modifications
4. Consider design changes if margins are insufficient
```

## Key Capabilities Demonstrated

### 1. Multi-Standard Compliance

- CISPR 32 (commercial)
- FCC Part 15 (regulatory)
- IEC 61000 (power quality)
- MIL-STD-461 (military)

**Single toolkit for all standards**

### 2. Automated Limit Checking

```python
compliance = check_compliance(
    frequencies, levels,
    limit_freq, limit_values,
    margin_db=6.0
)
```

**Features**:

- Interpolated limit lines
- Configurable margins
- Violation reporting
- Worst-case identification

### 3. Margin Analysis

**Engineering margins ensure production reliability**:

- Typical: 6 dB margin
- Critical: 10 dB margin
- Design goal: Maximum margin at minimum cost

### 4. Source Identification

**EMI fingerprinting identifies**:

- Switching frequencies
- Resonance peaks
- Digital clock harmonics
- RF sources (WiFi, cellular)

**Enables targeted fixes**

### 5. Comprehensive Reporting

**Test reports include**:

- Pass/fail status
- All violations
- Margin analysis
- Recommendations
- Traceability to standards

## Real-World Applications

### Consumer Electronics

**Requirements**: FCC Part 15 Class B, CISPR 32 Class B
**Challenges**:

- Tight PCB layouts
- High-speed digital circuits
- Cost constraints

**Oscura helps**:

- Pre-compliance testing
- Filter optimization
- Layout verification

### Medical Devices

**Requirements**: IEC 60601-1-2 (EMC for medical)
**Challenges**:

- Patient safety critical
- Harsh hospital environment
- Complex electronics

**Oscura helps**:

- Immunity testing
- Emission characterization
- Risk assessment

### Automotive

**Requirements**: CISPR 25, ISO 11452 (automotive EMC)
**Challenges**:

- Severe electrical noise
- Safety-critical systems
- Temperature extremes

**Oscura helps**:

- Conducted immunity
- Radiated immunity
- Power supply disturbance

### Military/Aerospace

**Requirements**: MIL-STD-461, DO-160 (avionics)
**Challenges**:

- Extremely strict limits
- Full spectrum coverage
- Detailed documentation

**Oscura helps**:

- CE101, CE102 conducted emissions
- RE101, RE102 radiated emissions
- CS101, CS114 conducted susceptibility

### Industrial

**Requirements**: IEC 61000-6-4 (industrial environment)
**Challenges**:

- Heavy machinery interference
- Long cable runs
- Three-phase power

**Oscura helps**:

- Harmonic analysis
- Voltage flicker
- Surge immunity

## Troubleshooting Common Issues

### Issue: Exceeding Conducted Emission Limits

**Symptoms**: Peaks in 150 kHz - 30 MHz range

**Causes**:

- Inadequate input filtering
- Poor PCB layout
- Common-mode noise

**Solutions**:

1. Add common-mode choke
2. Improve ground plane
3. Add differential-mode filtering
4. Shield noisy circuits

**Verification**:

```python
# Test with filter
after_signal = capture_with_filter()
results = analyze_conducted_emissions(after_signal, limits)
improvement = results["min_margin"] - before_margin
print(f"Filter effectiveness: {improvement:.1f} dB")
```

### Issue: Harmonic Current Violations

**Symptoms**: Exceeding IEC 61000-3-2 limits

**Causes**:

- Non-linear loads (rectifiers)
- No power factor correction
- Large capacitive loads

**Solutions**:

1. Active PFC (Power Factor Correction)
2. Passive filtering
3. Multiple smaller supplies

**Verification**:

```python
# Measure harmonics
harmonics = analyze_power_quality_harmonics(signal, limits)
print(f"THD: {harmonics['thd_percent']:.1f}%")
print(f"Violations: {harmonics['violations']}")
```

### Issue: Radiated Emission Peaks

**Symptoms**: Narrow peaks in radiated scan

**Causes**:

- Clock harmonics
- Unshielded cables
- Antenna effects (PCB traces)

**Solutions**:

1. Spread-spectrum clocking
2. Shielded enclosures
3. Cable filtering
4. PCB layout optimization

**Identification**:

```python
# Fingerprint to find source
fingerprint = emc_fingerprinting(signal)
switching_freq = fingerprint["inferred_switching_freq"]
print(f"Likely source: {switching_freq / 1e6:.1f} MHz oscillator")
```

## Advanced Topics

### Quasi-Peak vs Average Detection

CISPR standards require both:

**Quasi-Peak (QP)**:

- Simulates human perception of interference
- Charge/discharge time constants
- Higher for repetitive signals

**Average (AV)**:

- True average over time
- Lower than QP for pulsed signals
- Often 13 dB below QP limit

**Implementation** (requires specialized receiver or post-processing):

```python
# QP detection would require charge/discharge simulation
# Average is simple RBW filter + averaging
```

### Pre-Scan Techniques

**Fast scan for quick assessment**:

1. Wide RBW (1 MHz) for speed
2. Peak detector
3. Identify problem areas
4. Detailed scan only where needed

**Saves time in development**

### Margin Stacking

**Account for uncertainties**:

- Measurement: ±3 dB
- Production variation: ±3 dB
- Temperature: ±2 dB
- Aging: ±1 dB

**Total required margin**: ~9 dB

**Design for worst case**

### EMI Filter Design

**Staged approach**:

1. Measure baseline emissions
2. Calculate required attenuation
3. Design filter (LC stages)
4. Verify performance
5. Iterate as needed

**Oscura supports**:

- Before/after comparison
- Filter effectiveness
- Optimization

## Performance Notes

### Large FFT for Frequency Resolution

Conducted emissions testing needs fine resolution:

```python
# Good resolution for 150 kHz - 30 MHz
nfft = 65536  # ~3 kHz resolution at 200 MSa/s
freq, mag = fft(signal, nfft=nfft)
```

**Trade-off**: Resolution vs computation time

### Chunked Processing for Long Captures

```python
from oscura.analyzers.spectral import fft_chunked

# Process 1 second capture in chunks
freq, mag = fft_chunked(
    long_signal,
    chunk_size=1000000,
    window="flattop"
)
```

### Memory-Mapped Loading

```python
# Load large files efficiently
signal = osc.load("large_capture.wfm", mmap=True)
```

## See Also

- **[03_spectral_analysis](../../03_spectral_analysis/)** - FFT and spectral basics
- **[06_spectral_compliance](../06_spectral_compliance/)** - Additional spectral compliance
- **[07_mixed_signal](../07_mixed_signal/)** - Mixed-signal analysis
- **User Guide**: EMC/EMI testing workflows
- **Standards**:
  - CISPR 32:2015 (Multimedia equipment)
  - IEC 61000-3-2:2018 (Harmonic currents)
  - IEC 61000-3-3:2013 (Voltage fluctuations)
  - IEC 61000-4-2:2008 (ESD immunity)
  - FCC CFR 47 Part 15 (Radio frequency devices)
  - MIL-STD-461G (Military EMI)

## Estimated Time

- **Quick review**: 10 minutes
- **Full demo**: 15-20 minutes
- **Hands-on experimentation**: 30-45 minutes

---

**Note**: This demo uses synthetic data designed to illustrate compliance testing concepts. Real-world EMC testing requires calibrated equipment and proper test setups per applicable standards.
