# Signal Generation and Synthesis

**Create test signals, protocol sequences, and impairment simulations.**

This section contains 3 demonstrations showing how to generate synthetic signals, create protocol test sequences, and simulate real-world signal impairments. Essential for testing, validation, and development without physical hardware.

---

## Prerequisites

Before running these demonstrations, ensure you have:

- **Completed Getting Started** - Run `demonstrations/00_getting_started/` first
- **Understanding of Waveforms** - Familiarity with sine, square, triangle waves
- **Protocol Knowledge** - Basic understanding of UART, SPI, I2C (for protocol generation)
- **Signal Quality Concepts** - Jitter, noise, distortion fundamentals

Check your setup:

```bash
python demonstrations/00_getting_started/00_hello_world.py
# Should show: ✓ All measurements validated!
```

---

## Demonstrations

| Demo                            | Time       | Difficulty       | Focus                            |
| ------------------------------- | ---------- | ---------------- | -------------------------------- |
| 01_signal_builder_comprehensive | 20 min     | Intermediate     | Complete SignalBuilder API       |
| 02_protocol_generation          | 15 min     | Intermediate     | Generate protocol test sequences |
| 03_impairment_simulation        | 15 min     | Advanced         | Real-world signal impairments    |
| **Total**                       | **50 min** | **Intermediate** | **Signal generation**            |

---

## Learning Path

Complete these demonstrations in order for comprehensive coverage:

```
01_signal_builder_comprehensive.py    02_protocol_generation.py    03_impairment_simulation.py
              ↓                                  ↓                              ↓
All waveform types                  Protocol sequences             Real-world impairments
Basic to advanced                   UART/SPI/I2C/CAN              Jitter/noise/distortion
Multi-channel generation            Test patterns                  Performance testing
```

### Estimated Time: 50 minutes

---

## Key Concepts

This section teaches:

1. **Waveform Generation** - Sine, square, triangle, sawtooth, pulse, noise
2. **Protocol Synthesis** - Generate valid protocol sequences for testing
3. **Impairment Simulation** - Add jitter, noise, distortion, offset
4. **Multi-Channel Generation** - Create synchronized multi-channel signals
5. **Advanced Waveforms** - Chirp, multitone, modulated signals

---

## Running Demonstrations

### Option 1: Run Individual Demo

```bash
# From the project root
python demonstrations/17_signal_generation/01_signal_builder_comprehensive.py

# Or from the demo directory
cd demonstrations/17_signal_generation
python 01_signal_builder_comprehensive.py
```

### Option 2: Run All Signal Generation Demos

```bash
# From the project root
python demonstrations/17_signal_generation/01_signal_builder_comprehensive.py && \
python demonstrations/17_signal_generation/02_protocol_generation.py && \
python demonstrations/17_signal_generation/03_impairment_simulation.py
```

### Option 3: Validate All Demonstrations

```bash
# From the project root
python demonstrations/validate_all.py
```

---

## What You'll Learn

### Demo 01: Signal Builder Comprehensive

**File**: `01_signal_builder_comprehensive.py`

**Demonstrates**:

- Complete SignalBuilder API coverage
- All waveform types (sine, square, triangle, sawtooth, pulse, noise)
- Impairment simulation (jitter, noise, distortion, offset)
- Multi-channel generation
- Advanced waveforms (chirp, multitone, AM/FM modulation)

**What you'll do**:

1. Generate all basic waveform types
2. Create multi-channel synchronized signals
3. Generate advanced waveforms (chirp, multitone)
4. Add impairments to test signals
5. Use SignalBuilder for test fixture creation

**Capabilities**:

- `SignalBuilder.sine_wave` - Generate sine waves
- `SignalBuilder.square_wave` - Generate square waves
- `SignalBuilder.triangle_wave` - Generate triangle waves
- `SignalBuilder.sawtooth_wave` - Generate sawtooth waves
- `SignalBuilder.pulse_train` - Generate pulse sequences
- `SignalBuilder.white_noise` - Generate noise signals
- `SignalBuilder.chirp` - Generate frequency sweeps
- `SignalBuilder.multitone` - Generate multi-frequency signals

**IEEE Standards**: IEEE 1241-2010 (ADC Terminology)

**Related Demos**:

- `02_basic_analysis/01_waveform_measurements.py` - Measure generated signals
- `17_signal_generation/02_protocol_generation.py` - Protocol sequences
- `17_signal_generation/03_impairment_simulation.py` - Add impairments

---

### Demo 02: Protocol Generation

**File**: `02_protocol_generation.py`

**Demonstrates**:

- UART test sequence generation
- SPI transaction synthesis
- I2C protocol generation
- CAN frame creation
- Multi-protocol test patterns

**What you'll do**:

1. Generate UART byte sequences with correct framing
2. Create SPI transactions (MOSI/MISO/CLK/CS)
3. Generate I2C read/write sequences with ACK/NACK
4. Create CAN frames with proper CRC
5. Build multi-protocol test scenarios

**Capabilities**:

- UART frame generation (start/stop bits, parity)
- SPI transaction synthesis (modes 0-3)
- I2C sequence creation (address, data, ACK)
- CAN frame generation (standard/extended)
- Protocol timing accuracy

**Related Demos**:

- `03_protocol_decoding/01_serial_comprehensive.py` - Decode generated protocols
- `17_signal_generation/01_signal_builder_comprehensive.py` - Basic waveforms
- `18_comparison_testing/01_golden_reference.py` - Test against references

---

### Demo 03: Impairment Simulation

**File**: `03_impairment_simulation.py`

**Demonstrates**:

- Jitter simulation (random, periodic, systematic)
- Noise addition (white, pink, EMI)
- Harmonic distortion (THD)
- DC offset and drift
- Real-world signal degradation

**What you'll do**:

1. Add jitter to clock signals
2. Simulate different noise types
3. Add harmonic distortion to test robustness
4. Simulate DC offset and drift
5. Create realistic signal impairments for testing

**Capabilities**:

- Random jitter simulation
- Periodic jitter (sinusoidal)
- White noise addition
- Harmonic distortion (2nd, 3rd, total)
- DC offset and linear drift
- Combined impairment scenarios

**Related Demos**:

- `02_basic_analysis/03_filtering.py` - Filter impaired signals
- `04_advanced_analysis/02_power_analysis.py` - Analyze signal quality
- `17_signal_generation/01_signal_builder_comprehensive.py` - Base signals

---

## Troubleshooting

### "Generated signal has unexpected frequency"

**Solution**: Check sample rate and ensure Nyquist criterion is met:

```python
# Bad: Signal frequency too close to sample rate
signal = SignalBuilder.sine_wave(
    frequency=45000.0,
    sample_rate=50000.0  # Only 1.1x Nyquist - aliasing!
)

# Good: Sample rate ≥ 10x signal frequency
signal = SignalBuilder.sine_wave(
    frequency=1000.0,
    sample_rate=100000.0  # 100x Nyquist - clean signal
)
```

### "Protocol generation timing is incorrect"

**Solution**: Verify baud rate and sample rate relationship:

```python
# Ensure enough samples per bit
baud_rate = 9600
sample_rate = 1000000.0  # ~104 samples per bit ✓

# Too few samples causes timing errors
sample_rate = 10000.0    # ~1 sample per bit ✗
```

### "Impairment simulation crashes with NaN"

**Solution**: Check impairment magnitude is reasonable:

```python
# Bad: Noise magnitude exceeds signal
signal = add_noise(clean_signal, noise_level=10.0)  # Signal ±1V, noise 10V

# Good: Noise proportional to signal
signal = add_noise(clean_signal, noise_level=0.1)   # 10% noise
```

---

## Next Steps

### If You Want to...

| Goal                              | Next Demo                                         | Path               |
| --------------------------------- | ------------------------------------------------- | ------------------ |
| Test protocol decoders            | `03_protocol_decoding/01_serial_comprehensive.py` | Protocol decoding  |
| Compare against reference signals | `18_comparison_testing/01_golden_reference.py`    | Comparison testing |
| Measure signal quality            | `02_basic_analysis/01_waveform_measurements.py`   | Basic analysis     |
| Filter impaired signals           | `02_basic_analysis/03_filtering.py`               | Filtering          |

### Recommended Next Sections

1. **Comparison Testing** (18_comparison_testing/)
   - Test generated signals against references
   - Validate signal quality
   - Production testing workflows

2. **Protocol Decoding** (03_protocol_decoding/)
   - Decode generated protocol sequences
   - Validate protocol generators
   - Test decoder robustness

3. **Basic Analysis** (02_basic_analysis/)
   - Measure generated signal characteristics
   - Validate signal quality
   - Verify generation accuracy

---

## Understanding Signal Generation

### SignalBuilder Architecture

SignalBuilder is Oscura's single source of truth for test signal generation:

- **Deterministic** - Same parameters always produce same signal
- **IEEE-Compliant** - Follows IEEE 1241-2010 ADC terminology
- **Comprehensive** - All common waveforms and impairments
- **Test Fixture** - Used throughout Oscura's test suite

### Waveform Types

Different waveforms for different test scenarios:

| Waveform | Use Case                     | Key Parameters              |
| -------- | ---------------------------- | --------------------------- |
| Sine     | Frequency response, filters  | Frequency, amplitude, phase |
| Square   | Digital logic, timing        | Frequency, duty cycle       |
| Triangle | Linearity, slew rate         | Frequency, amplitude        |
| Sawtooth | Oscilloscope triggers, sweep | Frequency, rise/fall ratio  |
| Pulse    | Protocol testing, timing     | Width, period, amplitude    |
| Noise    | Noise floor, SNR testing     | Type (white/pink), level    |

### Impairment Types

Real-world signal degradations:

1. **Jitter** - Timing variations in clock signals
   - Random jitter (thermal noise)
   - Periodic jitter (power supply ripple)
   - Systematic jitter (clock distribution)

2. **Noise** - Amplitude variations
   - White noise (equal power at all frequencies)
   - Pink noise (1/f, decreases with frequency)
   - EMI (electromagnetic interference)

3. **Distortion** - Waveform shape changes
   - Harmonic distortion (non-linearity)
   - Clipping (amplitude limiting)
   - Ringing (underdamped response)

4. **Offset/Drift** - DC level changes
   - DC offset (constant shift)
   - Linear drift (temperature effects)
   - Random walk (low-frequency noise)

---

## Best Practices

### Signal Generation Strategy

1. **Start Clean** - Generate ideal signals first
2. **Add Impairments Gradually** - One impairment type at a time
3. **Validate Each Step** - Measure characteristics after each change
4. **Document Parameters** - Record all generation parameters

### Protocol Generation

1. **Follow Standards** - Use official protocol specifications
2. **Include Edge Cases** - Test boundary conditions
3. **Validate Timing** - Ensure protocol timing is accurate
4. **Test Decoder** - Verify generated protocols decode correctly

### Impairment Simulation

1. **Realistic Levels** - Use impairments matching real hardware
2. **Combined Effects** - Test multiple impairments together
3. **Worst Case** - Include maximum specified impairments
4. **Statistical Variation** - Use random impairments for Monte Carlo testing

---

## Advanced Techniques

### Multi-Channel Synchronized Signals

Generate phase-aligned multi-channel signals:

```python
# Create synchronized channels
builder = SignalBuilder(sample_rate=100000.0)

ch1 = builder.sine_wave(frequency=1000.0, phase=0.0)
ch2 = builder.sine_wave(frequency=1000.0, phase=90.0)  # 90° phase shift
ch3 = builder.sine_wave(frequency=1000.0, phase=180.0) # 180° phase shift

# Verify synchronization
assert ch1.metadata.sample_rate == ch2.metadata.sample_rate
```

### Custom Waveform Generation

Create arbitrary waveforms:

```python
# Define custom waveform function
def custom_wave(t):
    """Custom waveform: sum of harmonics."""
    return (np.sin(2*np.pi*1000*t) +
            0.5*np.sin(2*np.pi*2000*t) +
            0.25*np.sin(2*np.pi*3000*t))

# Generate using SignalBuilder
trace = SignalBuilder.custom_waveform(
    waveform_func=custom_wave,
    duration=0.1,
    sample_rate=100000.0
)
```

### Protocol Fuzzing

Generate invalid protocol sequences for robustness testing:

```python
# Generate valid UART sequence
valid_uart = generate_uart(data=[0x55, 0xAA])

# Add protocol errors for fuzzing
fuzzed = add_protocol_errors(
    valid_uart,
    framing_errors=0.1,    # 10% framing errors
    parity_errors=0.05,    # 5% parity errors
    break_conditions=True  # Include break conditions
)
```

### Impairment Profiles

Create realistic impairment profiles for different environments:

```python
# Industrial environment profile
industrial_profile = ImpairmentProfile(
    jitter_rms=50e-12,      # 50 ps RMS jitter
    noise_level=0.05,       # 5% noise
    emi_frequency=60.0,     # 60 Hz EMI
    harmonic_distortion=0.02 # 2% THD
)

# Apply profile to signal
impaired = industrial_profile.apply(clean_signal)
```

---

## Tips for Success

### Choose Appropriate Sample Rate

Higher sample rates for accurate generation:

```python
# For 1 kHz signal
sample_rate = 100000.0  # 100 samples per cycle ✓

# Minimum (Nyquist)
sample_rate = 2000.0    # 2 samples per cycle - aliasing!

# Recommended: 10-100x signal frequency
sample_rate = 10 * frequency_max
```

### Validate Generated Signals

Always verify generated signals meet specifications:

```python
# Generate signal
trace = SignalBuilder.sine_wave(frequency=1000.0)

# Validate
measured_freq = frequency(trace)
assert abs(measured_freq - 1000.0) < 1.0  # Within 1 Hz

measured_amp = amplitude(trace)
assert abs(measured_amp - 1.0) < 0.01  # Within 1%
```

### Use Appropriate Impairment Levels

Match impairments to real-world specifications:

```python
# Check datasheet for typical jitter
datasheet_jitter = 100e-12  # 100 ps

# Use datasheet values for realistic testing
signal = add_jitter(clean_signal, rms_jitter=datasheet_jitter)

# Use 3x for worst-case testing
worst_case = add_jitter(clean_signal, rms_jitter=3*datasheet_jitter)
```

---

## Protocol-Specific Generation

### UART Generation

```python
# Generate UART byte sequence
uart_signal = generate_uart(
    data=[0x48, 0x65, 0x6C, 0x6C, 0x6F],  # "Hello"
    baud_rate=9600,
    parity='N',      # None, 'E' (even), 'O' (odd)
    stop_bits=1,     # 1 or 2
    sample_rate=1000000.0
)
```

### SPI Generation

```python
# Generate SPI transaction
spi_signal = generate_spi(
    mosi_data=[0xAB, 0xCD, 0xEF],
    miso_data=[0x12, 0x34, 0x56],
    clock_frequency=1000000,  # 1 MHz
    mode=0,  # CPOL=0, CPHA=0
    sample_rate=10000000.0
)
```

### I2C Generation

```python
# Generate I2C write sequence
i2c_signal = generate_i2c(
    address=0x50,
    write_data=[0x00, 0x01, 0x02],
    clock_frequency=100000,  # 100 kHz standard mode
    sample_rate=10000000.0
)
```

---

## Summary

The Signal Generation section covers:

| Demo                            | Focus                        | Outcome                          |
| ------------------------------- | ---------------------------- | -------------------------------- |
| 01_signal_builder_comprehensive | Complete waveform generation | All signal types and impairments |
| 02_protocol_generation          | Protocol test sequences      | UART/SPI/I2C/CAN generation      |
| 03_impairment_simulation        | Real-world degradations      | Jitter/noise/distortion testing  |

After completing these 50-minute demonstrations, you'll understand:

- How to generate all common waveform types (sine, square, triangle, etc.)
- How to create protocol test sequences for UART, SPI, I2C, CAN
- How to simulate real-world signal impairments (jitter, noise, distortion)
- How to use SignalBuilder as a test fixture
- How to create multi-channel synchronized signals
- How to validate generated signal quality

**Ready to start?** Run this to explore signal generation:

```bash
python demonstrations/17_signal_generation/01_signal_builder_comprehensive.py
```

Happy signal generation!
