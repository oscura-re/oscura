# Side-Channel Analysis Guide

**Version**: 0.3.0
**Last Updated**: 2026-01-20

Guide to power analysis, timing attacks, and electromagnetic analysis using Oscura.

---

## Overview

Side-channel analysis exploits physical emissions from devices to extract secret information:

- **Power Analysis**: Current/voltage variations during computation
- **Timing Analysis**: Execution time variations
- **Electromagnetic Analysis**: EM emissions during operations
- **Acoustic Analysis**: Sound patterns from devices

**Use Cases**:

- Cryptographic key extraction
- Algorithm reverse engineering
- Security vulnerability assessment
- Device characterization
- Hardware trojan detection

---

## Power Analysis

### Simple Power Analysis (SPA)

Examine individual power traces to understand algorithm behavior.

```python
from oscura import load
from oscura.analyzers.power import analyze_power

# Load power trace from oscilloscope
trace = load("aes_encryption_power.wfm")

# Analyze power consumption
power_metrics = analyze_power(trace)

print(f"Average power: {power_metrics['average']:.3f} W")
print(f"Peak power: {power_metrics['peak']:.3f} W")
print(f"Power variation: {power_metrics['std']:.3f} W")

# Visualize
from oscura.visualization import plot_trace
plot_trace(trace, title="AES Encryption Power Trace")
```

---

### Differential Power Analysis (DPA)

Statistical attack correlating power consumption with data values.

```python
from oscura.sessions import BlackBoxSession
from oscura.acquisition import FileSource
import numpy as np

# Setup session for DPA
session = BlackBoxSession(name="AES Key Extraction")

# Collect many traces with known plaintexts
for i in range(1000):
    session.add_recording(
        f"plaintext_{i:04d}",
        FileSource(f"traces/aes_trace_{i:04d}.wfm")
    )

# Perform differential analysis
# Group traces by hypothetical key byte value
def dpa_attack(session, byte_position, hypothesis):
    """
    DPA attack on specific byte position.

    Args:
        session: Session with power traces
        byte_position: Byte position to attack (0-15 for AES)
        hypothesis: Key hypothesis function

    Returns:
        Correlation results for each key guess
    """
    correlations = np.zeros(256)

    for key_guess in range(256):
        # Partition traces based on hypothesis
        group_0 = []
        group_1 = []

        for name in session.list_recordings():
            trace = session.get_recording(name)
            plaintext = extract_plaintext(name)  # From filename

            # Hypothetical intermediate value
            intermediate = hypothesis(plaintext[byte_position], key_guess)

            # Partition by Hamming weight LSB
            if bin(intermediate).count('1') % 2 == 0:
                group_0.append(trace.data)
            else:
                group_1.append(trace.data)

        # Calculate differential trace
        avg_0 = np.mean(group_0, axis=0)
        avg_1 = np.mean(group_1, axis=0)
        diff = np.abs(avg_0 - avg_1)

        # Correlation peak indicates correct key
        correlations[key_guess] = np.max(diff)

    return correlations

# Attack each byte
aes_key = bytearray(16)
for byte_pos in range(16):
    correlations = dpa_attack(session, byte_pos, sbox_output)
    aes_key[byte_pos] = np.argmax(correlations)
    print(f"Byte {byte_pos}: 0x{aes_key[byte_pos]:02X}")

print(f"\nRecovered key: {aes_key.hex()}")
```

---

### Correlation Power Analysis (CPA)

More powerful variant using Pearson correlation coefficient.

```python
from scipy.stats import pearsonr

def cpa_attack(session, byte_position):
    """
    CPA attack using Pearson correlation.

    Returns:
        Best key guess and correlation coefficient
    """
    traces = []
    hamming_weights = []

    # Collect traces and hypothetical power values
    for key_guess in range(256):
        hw_values = []

        for name in session.list_recordings():
            trace = session.get_recording(name)
            plaintext = extract_plaintext(name)

            # Hypothetical intermediate value
            intermediate = sbox[plaintext[byte_position] ^ key_guess]

            # Hamming weight as power model
            hw = bin(intermediate).count('1')
            hw_values.append(hw)

        hamming_weights.append(hw_values)

    # Calculate correlation for each sample point
    best_correlation = 0
    best_key = 0

    for key_guess in range(256):
        for sample in range(len(traces[0])):
            power_values = [t[sample] for t in traces]
            correlation, _ = pearsonr(power_values, hamming_weights[key_guess])

            if abs(correlation) > abs(best_correlation):
                best_correlation = correlation
                best_key = key_guess

    return best_key, best_correlation

# Attack
key_byte, correlation = cpa_attack(session, 0)
print(f"Key byte 0: 0x{key_byte:02X} (correlation: {correlation:.3f})")
```

---

### Template Attacks

Build statistical templates for each operation.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def build_templates(session, operation):
    """
    Build power templates for specific operation.

    Args:
        session: Session with labeled traces
        operation: Operation name to template

    Returns:
        LDA model and templates
    """
    traces = []
    labels = []

    for name in session.list_recordings():
        if operation in name:
            trace = session.get_recording(name)
            label = extract_label(name)

            traces.append(trace.data)
            labels.append(label)

    # Train LDA model
    lda = LinearDiscriminantAnalysis()
    lda.fit(traces, labels)

    return lda

def template_attack(session, lda_model, unknown_trace):
    """
    Use templates to classify unknown trace.
    """
    prediction = lda_model.predict([unknown_trace.data])
    probabilities = lda_model.predict_proba([unknown_trace.data])

    return prediction[0], probabilities[0]

# Build templates from training data
lda = build_templates(session, "aes_sbox")

# Attack unknown trace
unknown = load("unknown_encryption.wfm")
key_byte, confidence = template_attack(session, lda, unknown)
print(f"Predicted key byte: 0x{key_byte:02X} (confidence: {max(confidence):.2%})")
```

---

## Timing Analysis

### Timing Attack on RSA

Exploit timing variations in modular exponentiation.

```python
from oscura import load
from oscura.analyzers.waveform import measure_pulse_width
import numpy as np

def timing_attack_rsa(traces_dir, public_key):
    """
    Timing attack on RSA private key.

    Args:
        traces_dir: Directory with timing traces
        public_key: (n, e) RSA public key

    Returns:
        Recovered private key bits
    """
    # Collect timing measurements
    timings = []
    messages = []

    for i in range(1000):
        trace = load(f"{traces_dir}/rsa_trace_{i:04d}.wfm")

        # Measure decryption time
        duration = measure_pulse_width(trace)
        timings.append(duration)

        # Extract message from filename metadata
        message = extract_message(f"rsa_trace_{i:04d}.wfm")
        messages.append(message)

    # Statistical analysis
    n, e = public_key
    key_bits = []

    for bit_pos in range(n.bit_length()):
        # Partition by bit value in message^e mod n
        group_0_times = []
        group_1_times = []

        for msg, time in zip(messages, timings):
            if (pow(msg, e, n) >> bit_pos) & 1:
                group_1_times.append(time)
            else:
                group_0_times.append(time)

        # Significant timing difference reveals key bit
        diff = abs(np.mean(group_1_times) - np.mean(group_0_times))
        if diff > threshold:
            key_bits.append(1 if np.mean(group_1_times) > np.mean(group_0_times) else 0)
        else:
            key_bits.append(0)  # Guess

    # Reconstruct private key
    private_key = sum(bit << i for i, bit in enumerate(key_bits))
    return private_key

# Perform attack
n = 0x9292758453063D803DD603D5E777D788  # Example modulus
e = 65537
d_recovered = timing_attack_rsa("captures/rsa/", (n, e))
print(f"Recovered private key: 0x{d_recovered:X}")
```

---

### Cache Timing Attacks

Detect cache hits/misses to infer secret data.

```python
def cache_timing_attack(session):
    """
    Cache timing attack on AES T-table implementation.
    """
    # Collect timing traces
    timing_data = {}

    for name in session.list_recordings():
        trace = session.get_recording(name)
        plaintext = extract_plaintext(name)

        # Measure timing for each cache line access
        cache_times = measure_cache_access_times(trace)
        timing_data[plaintext] = cache_times

    # Analyze timing variations
    # Cache hits are faster than misses
    for byte_pos in range(16):
        cache_access_pattern = {}

        for plaintext, times in timing_data.items():
            table_index = plaintext[byte_pos]
            if table_index not in cache_access_pattern:
                cache_access_pattern[table_index] = []
            cache_access_pattern[table_index].append(times[byte_pos])

        # Find access patterns that cluster (same cache line)
        for index, times in cache_access_pattern.items():
            if np.std(times) < threshold:
                print(f"Byte {byte_pos}, index {index}: likely same cache line")
```

---

## Electromagnetic Analysis

### EM Side-Channel Setup

```python
from oscura import load
from oscura.analyzers.spectral import analyze_spectrum

# Load EM trace from near-field probe
em_trace = load("em_capture_aes.wfm")

# Frequency domain analysis
spectrum = analyze_spectrum(em_trace)

# Find clock frequency
clock_freq = find_fundamental(spectrum)
print(f"Device clock: {clock_freq/1e6:.2f} MHz")

# Look for harmonic leakage
harmonics = find_harmonics(spectrum, clock_freq)
for i, (freq, power) in enumerate(harmonics):
    print(f"Harmonic {i+1}: {freq/1e6:.2f} MHz, {power:.1f} dBm")
```

---

### EM-Based DPA

```python
def em_dpa_attack(session, byte_position):
    """
    DPA attack using electromagnetic emissions.
    """
    # Similar to power DPA but using EM traces
    correlations = np.zeros(256)

    for key_guess in range(256):
        group_0_em = []
        group_1_em = []

        for name in session.list_recordings():
            em_trace = session.get_recording(name)
            plaintext = extract_plaintext(name)

            intermediate = sbox[plaintext[byte_position] ^ key_guess]

            if bin(intermediate).count('1') % 2 == 0:
                group_0_em.append(em_trace.data)
            else:
                group_1_em.append(em_trace.data)

        # EM differential
        avg_0 = np.mean(group_0_em, axis=0)
        avg_1 = np.mean(group_1_em, axis=0)
        diff = np.abs(avg_0 - avg_1)

        correlations[key_guess] = np.max(diff)

    return np.argmax(correlations)

# Attack using EM traces
for byte_pos in range(16):
    key_byte = em_dpa_attack(session, byte_pos)
    print(f"Byte {byte_pos}: 0x{key_byte:02X}")
```

---

## Advanced Techniques

### Trace Alignment

Align traces to compensate for timing jitter.

```python
from oscura.inference.alignment import align_global

def align_power_traces(session):
    """
    Align all traces to reference using cross-correlation.
    """
    # Use first trace as reference
    ref_name = session.list_recordings()[0]
    ref_trace = session.get_recording(ref_name)

    aligned_session = BlackBoxSession(name="Aligned Traces")

    for name in session.list_recordings():
        trace = session.get_recording(name)

        # Align to reference
        aligned = align_global(trace, ref_trace)

        aligned_session.add_recording(name, aligned)

    return aligned_session

# Align before DPA
aligned = align_power_traces(session)
key = dpa_attack(aligned, byte_position=0, hypothesis=sbox_output)
```

---

### Noise Reduction

Improve signal quality through averaging and filtering.

```python
from oscura.filtering import low_pass, moving_average

def preprocess_traces(session):
    """
    Apply noise reduction to all traces.
    """
    processed = BlackBoxSession(name="Filtered Traces")

    for name in session.list_recordings():
        trace = session.get_recording(name)

        # Low-pass filter to remove high-frequency noise
        filtered = low_pass(trace, cutoff=10e6)

        # Moving average for smoothing
        smoothed = moving_average(filtered, window_size=10)

        processed.add_recording(name, smoothed)

    return processed

# Preprocess before analysis
clean_session = preprocess_traces(session)
```

---

### Multi-Channel Analysis

Combine multiple side channels for improved attacks.

```python
def multi_channel_attack(power_session, em_session, byte_position):
    """
    Combined power and EM analysis.
    """
    # Correlations from power analysis
    power_corr = dpa_attack(power_session, byte_position, hypothesis)

    # Correlations from EM analysis
    em_corr = em_dpa_attack(em_session, byte_position)

    # Combined score (weighted average)
    combined = 0.6 * power_corr + 0.4 * em_corr

    return np.argmax(combined)
```

---

## Countermeasure Detection

### Identify Protection Mechanisms

```python
def detect_countermeasures(session):
    """
    Detect if device implements side-channel countermeasures.
    """
    traces = [session.get_recording(name) for name in session.list_recordings()]

    # 1. Check for random delays (temporal masking)
    trace_lengths = [len(t.data) for t in traces]
    if np.std(trace_lengths) > threshold:
        print("Detected: Random delay insertion")

    # 2. Check for power consumption randomization
    power_variations = [np.var(t.data) for t in traces]
    if np.mean(power_variations) > threshold:
        print("Detected: Power randomization (masking)")

    # 3. Check for dummy operations
    cross_correlation = np.corrcoef([t.data for t in traces])
    avg_correlation = np.mean(cross_correlation)
    if avg_correlation < threshold:
        print("Detected: Dummy operations or shuffling")

    # 4. Check for constant-time implementation
    execution_times = [measure_pulse_width(t) for t in traces]
    if np.std(execution_times) < 1e-9:  # < 1 ns variation
        print("Detected: Constant-time implementation")
```

---

## Practical Examples

### Example 1: AES Key Extraction

Complete DPA attack on AES implementation.

```python
from oscura.sessions import BlackBoxSession
from oscura.acquisition import FileSource

# Setup
session = BlackBoxSession(name="AES-128 Key Recovery")

# Load 1000 power traces
print("Loading power traces...")
for i in range(1000):
    session.add_recording(
        f"trace_{i:04d}",
        FileSource(f"aes_traces/power_{i:04d}.wfm")
    )

print(f"Loaded {len(session.list_recordings())} traces")

# Align traces
print("Aligning traces...")
aligned = align_power_traces(session)

# Preprocess
print("Filtering noise...")
clean = preprocess_traces(aligned)

# Attack each key byte
print("Performing DPA attack...")
recovered_key = bytearray(16)

for byte_pos in range(16):
    print(f"  Attacking byte {byte_pos}...", end=' ')
    correlations = dpa_attack(clean, byte_pos, sbox_output)
    recovered_key[byte_pos] = np.argmax(correlations)
    print(f"0x{recovered_key[byte_pos]:02X}")

print(f"\nRecovered AES-128 Key: {recovered_key.hex()}")

# Validate
known_key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
if recovered_key == known_key:
    print("SUCCESS: Key correctly recovered!")
else:
    print(f"FAILED: Expected {known_key.hex()}")
```

---

### Example 2: RSA Timing Attack

```python
from oscura import load
import numpy as np
from pathlib import Path

# Collect timing measurements
traces_dir = Path("rsa_traces")
timings = []
ciphertexts = []

print("Collecting timing data...")
for trace_file in sorted(traces_dir.glob("*.wfm")):
    trace = load(trace_file)
    duration = measure_pulse_width(trace)
    timings.append(duration)

    # Extract ciphertext from metadata
    ciphertext = int(trace_file.stem.split('_')[2], 16)
    ciphertexts.append(ciphertext)

print(f"Collected {len(timings)} timing measurements")

# Statistical attack
n = 0x9292758453063D803DD603D5E777D788
e = 65537

print("Performing timing analysis...")
recovered_bits = []

for bit_pos in range(n.bit_length()):
    # Partition by bit value
    slow_times = []
    fast_times = []

    for ct, time in zip(ciphertexts, timings):
        if (ct >> bit_pos) & 1:
            slow_times.append(time)
        else:
            fast_times.append(time)

    # Statistical test
    mean_diff = abs(np.mean(slow_times) - np.mean(fast_times))
    if mean_diff > 1e-9:  # 1 ns threshold
        recovered_bits.append(1 if np.mean(slow_times) > np.mean(fast_times) else 0)
    else:
        recovered_bits.append(0)

# Reconstruct private key
d = sum(bit << i for i, bit in enumerate(recovered_bits))
print(f"Recovered private key: 0x{d:X}")

# Verify
test_msg = 0x123456789ABCDEF
encrypted = pow(test_msg, e, n)
decrypted = pow(encrypted, d, n)

if decrypted == test_msg:
    print("SUCCESS: Private key verified!")
else:
    print("FAILED: Incorrect key recovery")
```

---

## Best Practices

### 1. Trace Collection

**Good practices**:

- Collect many traces (1000+ for DPA)
- Use stable power supply
- Minimize external interference
- Control temperature
- Document all conditions

```python
# Metadata tracking
metadata = {
    "device": "Target Board v1.2",
    "temperature": "25°C",
    "supply_voltage": "3.3V",
    "oscilloscope": "Tektronix MDO3024",
    "sample_rate": "1 GS/s",
    "bandwidth": "200 MHz",
    "date": "2026-01-20",
    "operator": "researcher1"
}

session.metadata = metadata
```

---

### 2. Signal Quality

```python
# Validate trace quality
def validate_trace_quality(trace):
    """Check if trace is usable."""
    # Check SNR
    from oscura.analyzers.spectral import snr
    signal_snr = snr(trace)

    if signal_snr < 20:  # 20 dB minimum
        print(f"WARNING: Low SNR ({signal_snr:.1f} dB)")
        return False

    # Check for saturation
    if np.max(np.abs(trace.data)) > 0.95:
        print("WARNING: Signal may be saturated")
        return False

    return True

# Validate before analysis
for name in session.list_recordings():
    trace = session.get_recording(name)
    if not validate_trace_quality(trace):
        print(f"Removing low-quality trace: {name}")
        session.remove_recording(name)
```

---

### 3. Statistical Validation

```python
from scipy import stats

def validate_attack_result(key_guess, correlations):
    """
    Validate DPA result with statistical tests.
    """
    # Check if peak is significant
    sorted_corr = np.sort(correlations)[::-1]
    peak_ratio = sorted_corr[0] / sorted_corr[1]

    if peak_ratio < 1.5:
        print("WARNING: Weak correlation peak (ambiguous result)")
        return False

    # T-test against random distribution
    t_stat, p_value = stats.ttest_1samp(correlations, np.mean(correlations))

    if p_value > 0.01:
        print("WARNING: Correlation not statistically significant")
        return False

    return True
```

---

## Related Documentation

- [Black-Box Analysis Guide](blackbox-analysis.md) - Protocol reverse engineering
- [Power Analysis API](../api/power-analysis.md) - Power measurement functions
- [Signal Integrity Guide](../../demos/15_signal_integrity/) - Signal quality analysis
- [Spectral Analysis](../api/analysis.md) - Frequency domain analysis

---

## Further Reading

**Academic Papers**:

- "Differential Power Analysis" - Kocher et al., 1999
- "Template Attacks" - Chari et al., 2002
- "Cache-Timing Attacks on AES" - Bernstein, 2005

**Standards**:

- ISO/IEC 17825: Testing methods for side-channel resistance
- FIPS 140-3: Security requirements for cryptographic modules

**Tools**:

- ChipWhisperer: Open-source side-channel platform
- Inspector: Side-channel analysis framework
- Riscure Inspector: Commercial SCA tool

---

**Security Note**: This guide is for educational and authorized security research only. Unauthorized side-channel attacks may violate laws. Always obtain proper authorization before testing.
