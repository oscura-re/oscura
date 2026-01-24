# Oscura Demonstrations: Troubleshooting Guide

Common issues and solutions when running the 112 demonstrations across all 19 categories.

## File Format Issues

### VCD Files Not Loading

**Symptom:** "Invalid VCD header" error or format not recognized
**Causes:** Incomplete file, wrong encoding, non-standard VCD format, corrupted file
**Solutions:**

- Verify file is complete (check file size > 0)
- Ensure UTF-8 encoding
- Try opening in text editor to check VCD header format
- Most demos generate synthetic data - no external files needed

### WAV Files Channel Mismatch

**Symptom:** "Channel count mismatch" or incorrect audio interpretation
**Causes:** Multi-channel WAV when single-channel expected, incorrect sample rate assumption
**Solutions:**

- Check WAV file channel count
- Verify sample rate matches expectations
- Use `load_all_channels()` for multi-channel files

### File Format Not Supported

**Symptom:** "UnsupportedFormatError: Format not recognized"
**Causes:** File extension doesn't match format, file corrupted, format truly unsupported
**Solutions:**

- Check file extension matches actual format
- Verify file isn't corrupted (check size, open in hex editor)
- Consult `00_getting_started/02_supported_formats.py` for supported formats list
- For custom formats, use binary loader API as shown in `01_data_loading/05_custom_binary.py`

---

## Installation and Dependency Issues

### ModuleNotFoundError for oscura or h5py

**Symptom:** "No module named 'oscura'" or "No module named 'h5py'"
**Causes:** Package not installed, wrong Python environment, incomplete installation
**Solutions:**

```bash
# Install oscura
pip install oscura
# OR with uv
uv add oscura

# Install optional dependencies
pip install oscura[all]
pip install h5py scipy matplotlib

# Verify installation
python -c "import oscura; print(oscura.__version__)"
```

### Python Version Too Old

**Symptom:** "Python 3.12+ required" error
**Causes:** Running on Python < 3.12
**Solutions:**

```bash
# Check Python version
python --version

# Install Python 3.12+ from python.org or use version manager
# pyenv: pyenv install 3.12.0 && pyenv local 3.12.0
# conda: conda create -n py312 python=3.12 && conda activate py312
```

### Missing Optional Dependencies

**Symptom:** Specific demos fail with import errors (h5py, scipy, reportlab, etc.)
**Causes:** Optional dependencies not installed
**Solutions:**

- Install all optional dependencies: `pip install oscura[all]`
- Or install specific packages as needed:
  - `pip install h5py` for HDF5 support
  - `pip install scipy` for signal processing
  - `pip install matplotlib` for visualization
  - `pip install reportlab` for PDF generation
  - `pip install jinja2` for HTML report generation

---

## Data and Signal Issues

### "Frequency detection failed"

**Symptom:** Frequency measurement returns NaN or unrealistic values
**Causes:** Signal too noisy, insufficient cycles, DC offset obscuring zero crossings, sample rate inadequate
**Solutions:**

1. Ensure signal has at least 2 complete cycles: `assert len(trace.data) >= 1000`
2. Check for DC offset: `trace_clean = trace.data - np.mean(trace.data)`
3. Verify sample rate is adequate: `sample_rate >= 10 * max_signal_frequency`
4. Use `period()` instead of `frequency()` for noisy signals
5. Apply filtering before frequency detection

### "FFT results show unexpected peaks"

**Symptom:** FFT has large spurious peaks, spectral leakage visible
**Causes:** Windowing issues, non-integer cycles in capture, DC offset not removed, inadequate sample rate
**Solutions:**

1. Apply appropriate window function: Use Hann or Hamming window for general signals
2. Check for spectral leakage: Ensure integer number of cycles in capture window
3. Remove DC offset: `trace.data -= np.mean(trace.data)`
4. Verify sample rate satisfies Nyquist: `sample_rate >= 2 * max_frequency`
5. Increase FFT size for better frequency resolution

### "Measurements vary between runs"

**Symptom:** Same signal produces different measurement results on repeated runs
**Causes:** Random noise, insufficient measurement window, floating-point precision
**Solutions:**

1. Check signal-to-noise ratio: `snr_value = snr(trace, signal_freq=1000.0)`
2. Use longer measurement windows for averaging
3. Apply appropriate filtering before measurement
4. Validate with known reference signals: `test_signal = generate_sine_wave(frequency=1000.0, amplitude=1.0)`
5. Use ensemble methods (multiple measurement techniques) for robustness

### Filter Introduces Unexpected Artifacts

**Symptom:** Filtered signal shows ringing, oscillation, or distortion
**Causes:** Filter order too high, cutoff frequency inappropriate, phase distortion
**Solutions:**

1. Check filter order isn't too high (reduces ringing): Start with order 4
2. Verify cutoff frequency is appropriate for signal content
3. Consider phase distortion: Use Bessel filter for linear phase response
4. Apply forward-backward filtering (`filtfilt`) for zero phase distortion
5. Use lower filter order with better design (Butterworth over Chebyshev for ringing)

---

## Protocol Decoding Issues

### "Baud rate detection failed"

**Symptom:** Auto baud rate detection returns 0 or unrealistic values
**Causes:** Insufficient edges, noisy signal, incorrect sample rate
**Solutions:**

1. Ensure capture has sufficient bit transitions (at least 100+ edges)
2. Check signal integrity: Clean edges, adequate SNR (>40 dB)
3. Manually specify baud rate if auto-detection fails
4. Use longer capture window for better statistics
5. Pre-filter signal to reduce noise before detection

### "Framing errors in UART decode"

**Symptom:** UART packets show framing errors, incorrect data
**Causes:** Timing mismatch, incorrect configuration, clock drift
**Solutions:**

1. Verify baud rate matches transmitter exactly
2. Check data bits, parity, stop bits configuration: `decode_uart(trace, data_bits=8, parity='N', stop_bits=1)`
3. Validate sample rate is 10x baud rate minimum
4. Look for clock drift in long captures
5. Check for noise causing bit errors

### "CAN frames show CRC errors"

**Symptom:** CAN decode fails with "CRC error" on valid-looking frames
**Causes:** Bit stuffing interpretation issue, signal integrity problem, bit timing
**Solutions:**

1. Verify CAN high/low voltage levels are correct
2. Check for proper bus termination (120Ω on each end)
3. Validate bit timing and sample point selection
4. Look for noise during arbitration causing bit errors
5. Ensure sample rate is high enough (10x bit rate minimum)

### "I2C decode misses ACK/NACK"

**Symptom:** I2C packets missing acknowledgment bits
**Causes:** Threshold issue, clock stretching, setup/hold time violations
**Solutions:**

1. Adjust logic threshold for proper HIGH/LOW detection
2. Check for clock stretching by slave devices (long LOW periods on clock)
3. Verify SDA setup/hold times meet I2C spec
4. Look for bus capacitance affecting edge timing
5. Check for pull-up resistor issues

### "Auto-detection returns wrong protocol"

**Symptom:** Protocol detection identifies protocol incorrectly
**Causes:** Ambiguous signal characteristics, insufficient data, multiple valid protocols possible
**Solutions:**

1. Manually specify protocol if known
2. Check confidence scores in detection results
3. Provide more context (baud rate, bus type, expected protocol)
4. Use longer capture for better fingerprinting
5. Look for protocol-specific markers (sync bytes, headers)

---

## Signal Processing Issues

### Filter Parameter Issues

**Symptom:** Filter order too high causing instability or excessive ringing
**Causes:** Over-aggressive filtering, inappropriate filter design
**Solutions:**

1. Start with lower filter order (order=4 is typical)
2. Try different filter types (Butterworth, Chebyshev, Bessel)
3. Use `filtfilt` (forward-backward) for zero-phase filtering
4. Adjust cutoff frequency appropriately

### Trigger Detection False Events

**Symptom:** Trigger detects noise as valid events
**Causes:** Threshold set too low, insufficient hysteresis, noise on signal
**Solutions:**

1. Increase hysteresis to reject noise
2. Use holdoff time to prevent re-triggering
3. Apply filtering before triggering
4. Combine multiple trigger conditions (AND/OR logic)

---

## Quality and Analysis Issues

### "Jitter measurement shows unexpected results"

**Symptom:** TIE/C2C jitter values seem incorrect or noisy
**Causes:** Insufficient edge transitions, noise affecting edge detection, improper threshold
**Solutions:**

1. Ensure sufficient edge transitions (100+ edges minimum)
2. Check for noise affecting edge detection
3. Validate threshold settings for edge detection
4. Consider filtering before jitter analysis
5. Verify clock is reasonably stable

### "Power factor calculation shows values > 1.0"

**Symptom:** Power factor measurement exceeds 1.0 (physically impossible)
**Causes:** Measurement window issues, phase alignment problems, harmonic effects
**Solutions:**

1. Ensure voltage and current traces are time-aligned
2. Check that measurement includes integer number of cycles
3. Validate phase angle calculation
4. Look for harmonics affecting measurements
5. Verify reactive power sign convention

### "Eye diagram appears distorted"

**Symptom:** Eye diagram is off-center, compressed, or malformed
**Causes:** Synchronization issue, incorrect unit interval, significant jitter
**Solutions:**

1. Verify unit interval (bit time) is correct
2. Check for clock recovery accuracy
3. Ensure sufficient data edges for overlay
4. Look for significant jitter affecting alignment
5. Validate trigger/sync settings

### "Pattern discovery finds spurious patterns"

**Symptom:** Too many patterns detected, many seem random
**Causes:** Noise in signal, insufficient minimum length threshold, low correlation threshold
**Solutions:**

1. Increase minimum pattern length threshold
2. Require higher repetition count for validation
3. Apply filtering to reduce noise
4. Use longer capture window for statistics
5. Adjust correlation threshold for stricter matching

### "Quality metrics differ from oscilloscope"

**Symptom:** SNR, THD, ENOB values don't match hardware measurements
**Causes:** Different FFT window, sampling method, fundamental frequency identification
**Solutions:**

1. Check FFT window function (Hann, Hamming vs rectangular)
2. Verify coherent sampling (integer cycles in window)
3. Compare fundamental frequency identification
4. Validate harmonic peak detection
5. Check for different ENOB calculation methods

---

## Performance and Parallelization Issues

### ProcessPoolExecutor Hangs on Windows

**Symptom:** Windows process pool seems to hang or doesn't produce results
**Causes:** Windows requires special handling for multiprocessing
**Solutions:**

```python
# Add this protection
if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor

    def process_file(filename):
        return result

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_file, filenames))
```

### Performance Worse with Parallelism

**Symptom:** Parallel processing slower than serial execution
**Causes:** Parallelism overhead exceeds benefits, task size too small
**Solutions:**

- Use ThreadPoolExecutor for I/O-bound operations (file loading)
- Use ProcessPoolExecutor for CPU-bound operations (FFT, filtering)
- Only parallelize if total processing time > 10 seconds per file
- Check actual speedup: `estimated_speedup = serial_time / (serial_time / num_cores + overhead)`

### Memory Issues with Large Batches

**Symptom:** Out of memory errors when processing many files
**Causes:** Loading all signals at once, high-memory analysis operations
**Solutions:**

```python
# Process in chunks
for i in range(0, len(files), chunk_size):
    chunk = files[i:i+chunk_size]
    results = process_batch(chunk)
    # Clean up between chunks
    del chunk, results
```

### ETA Calculation Unstable or Inaccurate

**Symptom:** Estimated time jumps around or is very wrong
**Causes:** Small sample size for averaging, first task much slower/faster
**Solutions:**

- Use moving average of recent processing times (not first sample)
- Calculate ETA from middle samples after warm-up
- Increase window size for stability

---

## Session and Batch Processing Issues

### "Cannot add recording to session"

**Symptom:** Error when trying to add recording to session
**Causes:** Duplicate name, invalid data source, session closed
**Solutions:**

```python
session = GenericSession(name="my_session")

# Use unique names
session.add_recording("baseline", source=trace_source)
session.add_recording("test_01", source=trace_source2)  # Different name

# Verify data is valid
assert trace_source is not None
assert hasattr(trace_source, 'data')
```

### Session Metadata Not Preserved

**Symptom:** Metadata added to session isn't available later
**Causes:** Not updating metadata, trying to access before setting
**Solutions:**

```python
# Set metadata when creating
session = GenericSession(
    name="protocol_analysis",
    description="Widget XYZ protocol RE"
)

# Update as needed
session.metadata["analyst"] = "Alice"
session.metadata["hypothesis"] = "Bytes 2-3 are command"
```

### Cannot Compare Recordings

**Symptom:** Error when comparing recordings in session
**Causes:** Incompatible trace types, recordings not loaded
**Solutions:**

- Both recordings must be same trace type (both WaveformTrace or both DigitalTrace)
- Ensure recordings are loaded before comparison

---

## Extensibility and Plugin Issues

### "ModuleNotFoundError" When Importing Plugin

**Symptom:** Custom plugin fails to import
**Causes:** Plugin not in Python path, missing `__init__.py`, import errors
**Solutions:**

```python
# Add to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Ensure __init__.py exists in all directories
# Use absolute imports
from oscura.plugins import BasePlugin
from oscura.core.types import WaveformTrace
```

### Custom Measurement Not in Registry

**Symptom:** Custom measurement not appearing in list
**Causes:** Not registered before querying, scope issue
**Solutions:**

```python
# Register FIRST
osc.register_measurement("my_measurement", my_function)

# Then query
measurements = osc.list_measurements()
assert "my_measurement" in measurements
```

### Plugin Fails Health Check

**Symptom:** Plugin loads but fails validation
**Causes:** Missing dependencies, version incompatibility, incomplete metadata
**Solutions:**

1. Check all required dependencies are installed
2. Verify plugin version compatible with Oscura version
3. Ensure all required methods are implemented
4. Verify metadata is complete

### Template Generation Fails

**Symptom:** Template generation produces error
**Causes:** Output directory permissions, file conflicts, invalid path
**Solutions:**

```python
# Use temp directory for testing
import tempfile
from pathlib import Path

output_dir = Path(tempfile.mkdtemp())
osc.generate_plugin_template("my_plugin", output_dir=output_dir)

# Verify directory is writable
assert output_dir.is_dir()
```

---

## Export and Visualization Issues

### "Export failed: permission denied"

**Symptom:** File export fails with permission error
**Causes:** Output directory not writable, file locked by another process
**Solutions:**

```bash
# Check permissions
ls -la output/

# Create output directory with proper permissions
mkdir -p output/exports
chmod 755 output/exports
```

### WaveDrom Output Is Empty

**Symptom:** WaveDrom timing diagram generates but has no content
**Causes:** Digital signal has no state transitions, incorrect format
**Solutions:**

- Ensure digital signal has high/low states with clear transitions
- Check signal data isn't all zeros or all ones

### Report Generation Requires Additional Dependencies

**Symptom:** Report generation fails with missing module
**Causes:** Optional report dependencies not installed
**Solutions:**

```bash
# For PDF reports
pip install reportlab

# For LaTeX-based reports
sudo apt-get install texlive-latex-base

# For HTML (usually included)
pip install jinja2
```

---

## Signal Generation Issues

### "Generated signal has unexpected frequency"

**Symptom:** Generated sine wave doesn't have expected frequency
**Causes:** Sample rate too low, Nyquist criterion violated
**Solutions:**

```python
# Bad: Signal frequency too close to Nyquist
signal = SignalBuilder.sine_wave(
    frequency=45000.0,
    sample_rate=50000.0  # Only 1.1x Nyquist
)

# Good: Sample rate >> 2x signal frequency
signal = SignalBuilder.sine_wave(
    frequency=1000.0,
    sample_rate=100000.0  # 100x Nyquist
)
```

### "Protocol generation timing is incorrect"

**Symptom:** Generated protocol has timing errors
**Causes:** Insufficient samples per bit, baud rate mismatch
**Solutions:**

- Ensure adequate samples per bit: `sample_rate >= 100 * baud_rate`
- Verify baud rate: `samples_per_bit = sample_rate / baud_rate`

### "Impairment simulation crashes with NaN"

**Symptom:** Adding impairments produces NaN values
**Causes:** Impairment magnitude too large
**Solutions:**

```python
# Bad: Noise exceeds signal
signal = add_noise(clean_signal, noise_level=10.0)

# Good: Noise proportional to signal (e.g., 10%)
signal = add_noise(clean_signal, noise_level=0.1)
```

---

## Comparison and Validation Issues

### "Reference comparison fails but signals look identical"

**Symptom:** Correlation coefficient is low despite visual similarity
**Causes:** DC offset mismatch, timing misalignment, scaling differences
**Solutions:**

```python
# Remove DC offset
test_signal = test_signal.data - np.mean(test_signal.data)
ref_signal = ref_signal.data - np.mean(ref_signal.data)

# Align signals
aligned_test = align_signals(test_signal, ref_signal)

# Now compare
correlation = compare_signals(aligned_test, ref_signal)
```

### "Limit test fails with borderline values"

**Symptom:** Test fails when measurement is at exact limit
**Causes:** No tolerance margin, floating-point precision issues
**Solutions:**

```python
# Bad: No tolerance
limit = 5.0
assert measurement == 5.0

# Good: Include tolerance
limit = 5.0
tolerance = 0.01  # 1% tolerance
assert abs(measurement - limit) <= tolerance
```

### "Mask testing shows false violations"

**Symptom:** Valid signals fail mask test
**Causes:** Scale/unit mismatch, coordinate system mismatch
**Solutions:**

- Verify mask and signal use same coordinate system
- Check voltage scale: `mask_voltage * signal.metadata.vertical_scale`
- Check time scale: `mask_time * signal.metadata.time_scale`

### "Regression detection too sensitive"

**Symptom:** Too many false alarms for regression detection
**Causes:** Threshold too tight, doesn't account for signal variability
**Solutions:**

```python
# Calculate signal variance
signal_std = np.std(baseline_signals, axis=0)

# Set threshold as 3-sigma
threshold = 3.0 * signal_std

# Use adaptive thresholds
if snr > 40:  # High SNR
    threshold = 0.01
else:  # Low SNR
    threshold = 0.10
```

---

## Standards Compliance Issues

### "Rise time measurement doesn't match oscilloscope"

**Symptom:** Measured rise time differs from scope reading
**Causes:** Different measurement points (20%-80% vs 10%-90% etc.)
**Solutions:**

```python
# IEEE 181 standard uses 10%-90%
rise_time_181 = rise_time(signal, lower=0.1, upper=0.9)

# Some scopes use 20%-80%
rise_time_alt = rise_time(signal, lower=0.2, upper=0.8)

# Ensure you're comparing same methodology
```

### "ADC measurements show unexpected noise floor"

**Symptom:** Noise floor or SNR measurement seems wrong
**Causes:** Non-coherent sampling, spectral leakage
**Solutions:**

```python
# Ensure coherent sampling (integer cycles)
cycles = (len(signal.data) * frequency) / sample_rate
assert cycles == int(cycles)  # Must be integer

# Verify coherent frequency
signal = generate_sine_wave(frequency=1000.0, sample_rate=100000.0)
# Not 997.3 Hz which would cause leakage
```

### "Power measurements differ from power meter"

**Symptom:** RMS or power measurements don't match professional meter
**Causes:** Different RMS calculation, different measurement window
**Solutions:**

```python
# Oscura uses true RMS (IEEE 1459 compliant)
rms_voltage = np.sqrt(np.mean(voltage_trace.data ** 2))

# Not averaging method (only works for sinusoids)
# Ensure measurement window covers integer cycles
```

### "Automotive PHY compliance test fails"

**Symptom:** PHY test reports failures
**Causes:** Signal conditioning, termination issues, measurement setup
**Solutions:**

- Verify differential signaling (P and N channels)
- Check 100Ω differential termination in place
- Calculate differential voltage correctly: `diff = ch_p - ch_n`

---

## Exploratory Analysis Issues

### "Cannot detect signal type"

**Symptom:** Signal detection returns uncertain or incorrect type
**Causes:** Signal too noisy, too short, unusual characteristics
**Solutions:**

```python
# Ensure adequate signal length
assert len(trace.data) >= 1000  # Minimum 1000 samples

# Apply noise reduction first
filtered = lowpass_filter(trace, cutoff=10000.0)

# Re-try characterization
characteristics = characterize_signal(filtered)
```

### "Pattern matching returns no results"

**Symptom:** Fuzzy matching finds no matching patterns
**Causes:** Tolerance too strict, pattern not present
**Solutions:**

```python
# Increase tolerance for noisy signals
matches = fuzzy_match(signal, pattern, tolerance=0.2)  # 20% tolerance

# Try different similarity metrics
matches = fuzzy_match(signal, pattern, metric="dtw")  # Dynamic time warping
matches = fuzzy_match(signal, pattern, metric="cosine")  # Scale invariant
```

### "Signal recovery produces artifacts"

**Symptom:** Recovered signal has distortions or noise
**Causes:** Recovery parameters too aggressive
**Solutions:**

```python
# Use gentler filtering
recovered = recover_signal(corrupted, filter_strength=0.3)

# Try different interpolation methods
recovered = recover_signal(corrupted, interpolation="cubic")
recovered = recover_signal(corrupted, interpolation="spline")
```

---

## General Troubleshooting Steps

### When Any Demo Fails

1. **Check Python Version**

   ```bash
   python --version  # Must be 3.12+
   ```

2. **Verify Installation**

   ```bash
   python -c "import oscura; print(oscura.__version__)"
   ```

3. **Run Installation Validation**

   ```bash
   python demonstrations/validate_all.py
   ```

4. **Check for Typos and Common Mistakes**
   - Incorrect file paths
   - Wrong variable names
   - Missing imports
   - Indentation errors

5. **Enable Debug Output**

   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

6. **Test with Synthetic Data First**
   - Don't load external files initially
   - Use `generate_sine_wave()` to validate analysis code
   - Build up complexity gradually

7. **Read Demo Docstrings**

   ```python
   import demonstrations.category.demo_name as demo
   help(demo)  # Show full documentation
   ```

8. **Check Related Issues**
   - Look for similar issues in other demos
   - Check if it's a known limitation
   - Review issue database or discussions

9. **Validate Data Quality**

   ```python
   # Check trace properties
   print(f"Samples: {len(trace.data)}")
   print(f"Sample rate: {trace.metadata.sample_rate}")
   print(f"Duration: {len(trace.data) / trace.metadata.sample_rate} s")
   print(f"Data range: {np.min(trace.data)} to {np.max(trace.data)}")
   ```

10. **Compare Against Known-Good Output**
    - Run demo with test data
    - Compare output format and structure
    - Identify deviations

---

## Getting Additional Help

### Resources

- **Demonstrations**: See individual README.md files in each category
- **API Documentation**: Run `oscura.help()` or check docs/
- **Example Code**: Check `examples/` directory
- **Test Cases**: Review `tests/` for expected behavior
- **Community**: Check project issues and discussions

### Effective Bug Reports

When reporting issues, include:

1. **Exact Error Message** - Full traceback
2. **Python Version** - `python --version`
3. **Oscura Version** - `python -c "import oscura; print(oscura.__version__)"`
4. **Minimal Reproduction** - Smallest code that reproduces issue
5. **Expected Behavior** - What should happen
6. **Actual Behavior** - What actually happened
7. **Steps to Reproduce** - Exact sequence to reproduce

---

## Summary

This troubleshooting guide covers:

- **File Format Issues** - Loading, format detection, corruption
- **Installation Issues** - Dependencies, Python version, setup
- **Signal Processing** - Frequency detection, filtering, measurement
- **Protocol Decoding** - Baud rate, framing, CAN/I2C/UART
- **Analysis Quality** - Jitter, power, eye diagrams, patterns
- **Performance** - Parallelization, memory, ETA
- **Sessions & Batch** - Recording management, comparison
- **Extensibility** - Plugins, measurements, algorithms
- **Export & Visualization** - File formats, reports, diagrams
- **Signal Generation** - Waveforms, protocols, impairments
- **Validation** - Reference comparison, limits, masks, regression
- **Standards** - IEEE 181, 1241, 1459, 2414 compliance
- **Exploratory Analysis** - Unknown signals, pattern matching, recovery

**Most Common Issues:**

1. Old Python version (< 3.12)
2. Missing optional dependencies
3. Signal too noisy or too short
4. Incorrect signal parameters (sample rate, frequency)
5. Timing/alignment issues in protocol decoding
6. File format compatibility

For additional troubleshooting, see individual category READMEs for specific demo guidance.

Happy troubleshooting!
