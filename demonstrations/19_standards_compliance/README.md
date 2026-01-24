# IEEE Standards Compliance

**Validate measurements against IEEE standards for pulse, ADC, power, and PHY testing.**

This section contains 4 demonstrations showing compliance with IEEE 181 (pulse measurements), IEEE 1241 (ADC characterization), IEEE 1459 (power quality), and IEEE 2414 (automotive PHY). Essential for standards-compliant testing and industry-standard measurement validation.

---

## Prerequisites

Before running these demonstrations, ensure you have:

- **Completed Basic Analysis** - Run `demonstrations/02_basic_analysis/` first
- **Understanding of Standards** - Familiarity with IEEE standards concepts
- **Measurement Knowledge** - Rise time, fall time, THD, RMS concepts
- **Domain Expertise** - Understanding of relevant domain (power, automotive, etc.)

Check your setup:

```bash
python demonstrations/02_basic_analysis/01_waveform_measurements.py
# Should show successful measurements
```

---

## Demonstrations

| Demo         | Time       | Difficulty   | Focus                               |
| ------------ | ---------- | ------------ | ----------------------------------- |
| 01_ieee_181  | 15 min     | Intermediate | IEEE 181-2011 pulse measurements    |
| 02_ieee_1241 | 15 min     | Advanced     | IEEE 1241-2010 ADC characterization |
| 03_ieee_1459 | 15 min     | Advanced     | IEEE 1459-2010 power quality        |
| 04_ieee_2414 | 15 min     | Advanced     | IEEE 2414-2020 automotive PHY       |
| **Total**    | **60 min** | **Advanced** | **Standards compliance**            |

---

## Learning Path

Complete these demonstrations based on your domain expertise:

```
         IEEE Standards Compliance
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
01_ieee_181.py  02_ieee_1241.py  03_ieee_1459.py  04_ieee_2414.py
    ↓               ↓               ↓               ↓
Pulse testing   ADC testing     Power quality   Automotive PHY
General use     Instrumentation Manufacturing   Automotive
```

### Estimated Time: 60 minutes (select relevant standards)

---

## Key Concepts

This section teaches:

1. **IEEE 181** - Standard pulse measurement methods (rise/fall time, overshoot)
2. **IEEE 1241** - ADC characterization and dynamic performance
3. **IEEE 1459** - Power quality measurements (harmonics, distortion)
4. **IEEE 2414** - Automotive Ethernet PHY compliance testing
5. **Standards-Compliant Methodology** - Following official measurement procedures

---

## Running Demonstrations

### Option 1: Run Individual Demo

```bash
# From the project root
python demonstrations/19_standards_compliance/01_ieee_181.py

# Or from the demo directory
cd demonstrations/19_standards_compliance
python 01_ieee_181.py
```

### Option 2: Run All Standards Compliance Demos

```bash
# From the project root
python demonstrations/19_standards_compliance/01_ieee_181.py && \
python demonstrations/19_standards_compliance/02_ieee_1241.py && \
python demonstrations/19_standards_compliance/03_ieee_1459.py && \
python demonstrations/19_standards_compliance/04_ieee_2414.py
```

### Option 3: Validate All Demonstrations

```bash
# From the project root
python demonstrations/validate_all.py
```

---

## What You'll Learn

### Demo 01: IEEE 181 - Pulse Measurements

**File**: `01_ieee_181.py`

**Standard**: IEEE 181-2011 (Transitions, Pulses, and Related Waveforms)

**Demonstrates**:

- Rise/fall time measurement (10%-90%)
- Overshoot and undershoot measurement
- Pulse width and duty cycle
- Standard-compliant measurement methodology

**What you'll do**:

1. Measure rise time using IEEE 181 method (10%-90%)
2. Measure fall time using standard methodology
3. Detect and quantify overshoot/undershoot
4. Calculate pulse width and duty cycle
5. Validate measurements against standard requirements

**Capabilities**:

- `oscura.rise_time` - 10%-90% rise time (IEEE 181)
- `oscura.fall_time` - 90%-10% fall time (IEEE 181)
- `oscura.overshoot` - Overshoot percentage
- `oscura.undershoot` - Undershoot percentage
- `oscura.duty_cycle` - Duty cycle percentage

**Related Demos**:

- `02_basic_analysis/01_waveform_measurements.py` - Waveform measurements
- `18_comparison_testing/03_mask_testing.py` - Mask testing
- `17_signal_generation/01_signal_builder_comprehensive.py` - Generate test pulses

**Why This Matters**:
IEEE 181 defines industry-standard methods for measuring digital signal transitions. Compliance ensures measurements are comparable across different tools and organizations.

---

### Demo 02: IEEE 1241 - ADC Characterization

**File**: `02_ieee_1241.py`

**Standard**: IEEE 1241-2010 (Analog-to-Digital Converter Characterization)

**Demonstrates**:

- SNR (Signal-to-Noise Ratio) measurement
- SINAD (Signal-to-Noise and Distortion)
- ENOB (Effective Number of Bits)
- THD (Total Harmonic Distortion)
- SFDR (Spurious-Free Dynamic Range)

**What you'll do**:

1. Characterize ADC dynamic performance
2. Measure SNR using IEEE 1241 methodology
3. Calculate SINAD and derive ENOB
4. Measure THD and identify harmonic components
5. Calculate SFDR for spurious signal detection

**Capabilities**:

- SNR measurement (coherent sampling)
- SINAD calculation
- ENOB derivation from SINAD
- THD measurement (up to 9th harmonic)
- SFDR calculation

**Related Demos**:

- `04_advanced_analysis/01_spectral_analysis.py` - Spectral analysis
- `17_signal_generation/01_signal_builder_comprehensive.py` - Test signals
- `18_comparison_testing/02_limit_testing.py` - Limit testing

**Why This Matters**:
IEEE 1241 is the definitive standard for ADC testing. Understanding these measurements is critical for instrumentation design and validation.

---

### Demo 03: IEEE 1459 - Power Quality

**File**: `03_ieee_1459.py`

**Standard**: IEEE 1459-2010 (Power Definitions for Single-Phase and Polyphase Systems)

**Demonstrates**:

- RMS voltage and current
- Active, reactive, and apparent power
- Power factor and displacement power factor
- Total Harmonic Distortion (THD)
- Harmonic power contributions

**What you'll do**:

1. Measure RMS voltage and current per IEEE 1459
2. Calculate active power (watts)
3. Calculate reactive power (VARs)
4. Calculate apparent power (VA)
5. Measure power quality metrics (THD, power factor)

**Capabilities**:

- RMS measurements (true RMS)
- Active power calculation
- Reactive power calculation
- Apparent power calculation
- Power factor measurement
- THD calculation

**Related Demos**:

- `04_advanced_analysis/02_power_analysis.py` - Power analysis
- `05_domain_specific/01_power_quality.py` - Power quality testing
- `02_basic_analysis/01_waveform_measurements.py` - RMS measurements

**Why This Matters**:
IEEE 1459 standardizes power measurements for modern power systems with non-sinusoidal waveforms. Critical for power quality analysis and renewable energy systems.

---

### Demo 04: IEEE 2414 - Automotive Ethernet PHY

**File**: `04_ieee_2414.py`

**Standard**: IEEE 2414-2020 (Automotive Ethernet PHY Test)

**Demonstrates**:

- 100BASE-T1 PHY compliance testing
- Eye diagram measurements
- Jitter analysis
- Return loss measurement
- Differential voltage and timing

**What you'll do**:

1. Test automotive Ethernet PHY compliance
2. Measure eye diagram parameters
3. Analyze jitter (random and deterministic)
4. Validate differential voltage levels
5. Check timing parameters against standard

**Capabilities**:

- Eye diagram analysis
- Jitter measurement (RJ, DJ, TJ)
- Differential voltage measurement
- Return loss calculation
- Compliance pass/fail determination

**Related Demos**:

- `05_domain_specific/02_automotive.py` - Automotive testing
- `03_protocol_decoding/02_automotive.py` - CAN/LIN decoding
- `18_comparison_testing/03_mask_testing.py` - Mask testing

**Why This Matters**:
IEEE 2414 defines compliance testing for automotive Ethernet, the backbone of modern vehicle networks. Critical for automotive electronics development.

---

## Troubleshooting

### "Rise time measurement doesn't match oscilloscope"

**Solution**: Verify you're using the same measurement points (10%-90%):

```python
# IEEE 181 standard: 10%-90%
rise_time_181 = rise_time(signal, lower=0.1, upper=0.9)

# Some scopes use 20%-80%
rise_time_alt = rise_time(signal, lower=0.2, upper=0.8)

# Ensure you're comparing same methodology
```

### "ADC measurements show unexpected noise floor"

**Solution**: Ensure coherent sampling for IEEE 1241 compliance:

```python
# Bad: Non-coherent sampling causes spectral leakage
signal = generate_sine_wave(frequency=997.3, sample_rate=100000.0)

# Good: Coherent sampling (integer cycles in record)
signal = generate_sine_wave(frequency=1000.0, sample_rate=100000.0)

# Verify coherence
cycles = (len(signal.data) * frequency) / sample_rate
assert cycles == int(cycles)  # Must be integer
```

### "Power measurements differ from power meter"

**Solution**: Verify RMS calculation method (true RMS vs averaging):

```python
# True RMS (IEEE 1459 compliant)
rms_voltage = np.sqrt(np.mean(voltage_trace.data ** 2))

# Some meters use averaging (incorrect for non-sinusoidal)
# Oscura always uses true RMS for compliance
```

### "Automotive PHY compliance test fails"

**Solution**: Check test setup and signal conditioning:

```python
# Verify differential signaling
assert voltage_p.metadata.channel_name.endswith('_P')
assert voltage_n.metadata.channel_name.endswith('_N')

# Calculate differential voltage
diff_voltage = voltage_p.data - voltage_n.data

# Apply proper termination in test setup (100Ω differential)
```

---

## Next Steps

### If You Want to...

| Goal                                  | Next Demo                                                 | Path                  |
| ------------------------------------- | --------------------------------------------------------- | --------------------- |
| Apply standards to production testing | `16_complete_workflows/02_production_testing.py`          | Production workflows  |
| Generate compliant test signals       | `17_signal_generation/01_signal_builder_comprehensive.py` | Signal generation     |
| Compare against limits                | `18_comparison_testing/02_limit_testing.py`               | Limit testing         |
| Build domain-specific workflows       | `05_domain_specific/`                                     | Domain-specific demos |

### Recommended Next Sections

1. **Complete Workflows** (16_complete_workflows/)
   - Apply standards to real workflows
   - Production testing automation
   - Case studies

2. **Domain-Specific** (05_domain_specific/)
   - Power quality analysis (IEEE 1459)
   - Automotive testing (IEEE 2414)
   - Industry applications

3. **Comparison Testing** (18_comparison_testing/)
   - Test against specification limits
   - Validate compliance
   - Automated testing

---

## Understanding IEEE Standards

### IEEE 181 - Pulse Measurements

**Key Concepts**:

- **Rise Time** - Time from 10% to 90% of amplitude (not 0% to 100%)
- **Fall Time** - Time from 90% to 10% of amplitude
- **Overshoot** - Peak excursion above 100% level
- **Undershoot** - Peak excursion below 0% level

**Why 10%-90%**:

- Avoids noise at signal edges
- Provides repeatable measurements
- Industry-standard methodology

**Common Variations**:

- 20%-80% (some scopes, less sensitive to noise)
- 0%-100% (theoretical, not practical)

---

### IEEE 1241 - ADC Testing

**Key Metrics**:

| Metric | Formula                             | Meaning                          |
| ------ | ----------------------------------- | -------------------------------- |
| SNR    | 20·log₁₀(signal/noise)              | Signal quality without harmonics |
| SINAD  | 20·log₁₀(signal/(noise+distortion)) | Overall signal quality           |
| ENOB   | (SINAD - 1.76) / 6.02               | Effective resolution             |
| THD    | 20·log₁₀(√(H₂²+H₃²+...) / H₁)       | Harmonic distortion              |
| SFDR   | Signal power - Largest spur         | Spurious-free range              |

**Testing Requirements**:

- **Coherent Sampling** - Integer number of cycles in record
- **High Sample Rate** - ≥2.5× Nyquist for accuracy
- **Low Jitter** - Clock jitter < 1ps RMS for high-resolution ADCs

---

### IEEE 1459 - Power Quality

**Power Definitions**:

| Power Type | Symbol | Unit      | Formula        |
| ---------- | ------ | --------- | -------------- |
| Active     | P      | Watts (W) | ∫ v(t)·i(t) dt |
| Reactive   | Q      | VARs      | √(S² - P²)     |
| Apparent   | S      | VA        | Vrms · Irms    |

**Power Factor**:

- **True PF** = P / S (includes harmonics)
- **Displacement PF** = cos(φ) (fundamental only)

**Why This Matters**:
Non-sinusoidal waveforms (switching power supplies, VFDs) require IEEE 1459 definitions. Traditional power formulas fail with harmonics.

---

### IEEE 2414 - Automotive Ethernet

**100BASE-T1 Parameters**:

| Parameter            | Specification     | Test Method               |
| -------------------- | ----------------- | ------------------------- |
| Differential Voltage | 1.0V ± 10%        | Peak-to-peak measurement  |
| Rise/Fall Time       | ≤ 5ns             | 20%-80% method            |
| Random Jitter        | ≤ 400ps           | Statistical analysis      |
| Deterministic Jitter | ≤ 600ps           | Pattern-based measurement |
| Return Loss          | ≥ 10dB @ 66.67MHz | S-parameter measurement   |

**Eye Diagram Requirements**:

- Minimum eye height: 400mV
- Minimum eye width: 10ns
- Mask compliance required

---

## Best Practices

### Standards Compliance Testing

1. **Read the Standard** - Always reference the official standard document
2. **Follow Methodology Exactly** - Standards define specific procedures
3. **Document Deviations** - If you must deviate, document why
4. **Calibrate Equipment** - Use calibrated instruments with traceable standards
5. **Report Uncertainty** - Include measurement uncertainty in results

### Measurement Validation

1. **Compare to Reference** - Validate against known-good measurements
2. **Cross-Check Methods** - Use multiple measurement techniques
3. **Peer Review** - Have results reviewed by domain experts
4. **Instrument Comparison** - Compare against commercial instruments

### Documentation

1. **Test Conditions** - Document temperature, humidity, setup
2. **Equipment List** - Record all instruments with cal dates
3. **Procedure** - Document exact test procedure
4. **Results** - Include raw data, not just pass/fail
5. **Traceability** - Link to standard version and sections

---

## Advanced Topics

### Coherent Sampling for IEEE 1241

Ensure integer number of cycles for FFT-based measurements:

```python
# Calculate coherent frequency
sample_rate = 100000.0
num_samples = 10000
cycles = 17  # Integer number of cycles

coherent_freq = (cycles * sample_rate) / num_samples
# Result: 170.0 Hz (exactly 17 cycles in 10000 samples)

# Generate coherent signal
signal = generate_sine_wave(
    frequency=coherent_freq,
    sample_rate=sample_rate,
    num_samples=num_samples
)

# Verify coherence
actual_cycles = (num_samples * coherent_freq) / sample_rate
assert actual_cycles == cycles
```

### Harmonic Analysis for IEEE 1459

Calculate individual harmonic contributions:

```python
# Perform FFT
fft_result = fft_analysis(signal)

# Extract harmonics (up to 50th)
harmonics = []
fundamental_freq = 60.0  # 60 Hz power line

for n in range(1, 51):
    harmonic_freq = n * fundamental_freq
    harmonic_power = get_power_at_frequency(fft_result, harmonic_freq)
    harmonics.append({
        'order': n,
        'frequency': harmonic_freq,
        'power': harmonic_power
    })

# Calculate THD per IEEE 1459
fundamental_power = harmonics[0]['power']
harmonic_sum = sum(h['power'] for h in harmonics[1:])
thd = np.sqrt(harmonic_sum) / fundamental_power
```

### Eye Diagram Analysis for IEEE 2414

Generate and analyze eye diagrams:

```python
# Create eye diagram from serial data
eye = create_eye_diagram(
    signal,
    bit_rate=100e6,  # 100 Mbps
    samples_per_ui=100  # Unit Interval
)

# Measure eye parameters
eye_height = measure_eye_height(eye)
eye_width = measure_eye_width(eye)
eye_jitter = measure_eye_jitter(eye)

# Check compliance
assert eye_height >= 400e-3  # 400mV minimum
assert eye_width >= 10e-9    # 10ns minimum
assert eye_jitter <= 1e-9    # 1ns maximum total jitter
```

---

## Tips for Success

### Understand the Standard's Intent

Don't just follow formulas blindly:

```python
# IEEE 181: Why 10%-90%?
# Answer: Avoids noise at edges while capturing transition

# IEEE 1241: Why coherent sampling?
# Answer: Eliminates spectral leakage in FFT

# IEEE 1459: Why true RMS?
# Answer: Handles non-sinusoidal waveforms correctly

# IEEE 2414: Why specific jitter limits?
# Answer: Ensures BER < 10⁻¹⁰ for reliable communication
```

### Validate Test Setup

Ensure your test setup matches standard requirements:

```python
# Example: IEEE 2414 differential measurement
# Standard requires 100Ω differential termination

# Check setup
assert probe_impedance_p == 50  # Ω
assert probe_impedance_n == 50  # Ω
assert differential_impedance == 100  # Ω

# Verify differential signaling
diff_signal = channel_p.data - channel_n.data
common_mode = (channel_p.data + channel_n.data) / 2
assert np.max(np.abs(common_mode)) < 0.1 * np.max(np.abs(diff_signal))
```

### Keep Updated on Standards

Standards evolve - track revisions:

```python
# Document standard version in results
metadata = {
    'standard': 'IEEE 181',
    'version': '2011',
    'test_date': '2024-01-23',
    'implementation': 'Oscura v0.5.0'
}

# Check for newer versions periodically
# IEEE 181: 2003 → 2011 (current)
# IEEE 1241: 2000 → 2010 (current)
# IEEE 1459: 2000 → 2010 (current)
# IEEE 2414: 2020 (current)
```

---

## Summary

The IEEE Standards Compliance section covers:

| Demo         | Standard       | Focus                | Outcome                          |
| ------------ | -------------- | -------------------- | -------------------------------- |
| 01_ieee_181  | IEEE 181-2011  | Pulse measurements   | Rise/fall time, overshoot        |
| 02_ieee_1241 | IEEE 1241-2010 | ADC characterization | SNR, SINAD, ENOB, THD            |
| 03_ieee_1459 | IEEE 1459-2010 | Power quality        | Active/reactive power, THD       |
| 04_ieee_2414 | IEEE 2414-2020 | Automotive PHY       | Eye diagrams, jitter, compliance |

After completing these 60-minute demonstrations, you'll understand:

- How to perform IEEE 181 compliant pulse measurements
- How to characterize ADCs using IEEE 1241 methodology
- How to measure power quality per IEEE 1459 standards
- How to test automotive Ethernet PHY per IEEE 2414
- How to document standards-compliant measurements
- How to validate compliance and generate reports

**Ready to start?** Run this to explore IEEE 181 pulse measurements:

```bash
python demonstrations/19_standards_compliance/01_ieee_181.py
```

Happy standards-compliant testing!
