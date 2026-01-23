# Getting Started with Oscura

**Your introduction to the Oscura hardware reverse engineering framework.**

This section contains 3 foundational demonstrations designed to validate your installation and teach the core concepts you'll need for all advanced usage. Perfect for first-time users.

---

## Prerequisites

Before running these demonstrations, ensure you have:

- **Python 3.12+** - Oscura requires Python 3.12 or higher
- **Oscura installed** - Install with `pip install oscura` or `uv add oscura`
- **Basic familiarity with signals** - Understanding of analog/digital signals helps but isn't required
- **Terminal or Python IDE** - Any standard Python environment works

Check your Python version:

```bash
python --version
# Output should be: Python 3.12.x or higher
```

---

## Learning Path

These demonstrations are designed to be completed **in order**. Each builds on concepts from the previous one:

```
00_hello_world.py        01_core_types.py         02_supported_formats.py
        ↓                        ↓                           ↓
Validate installation    Understand data structures    Explore file formats
Learn basic API          Work with traces & metadata   See what's supported
```

### Estimated Time

| Demo                 | Time       | Difficulty   | Topics                                      |
| -------------------- | ---------- | ------------ | ------------------------------------------- |
| 00_hello_world       | 5 min      | Beginner     | Installation validation, basic measurements |
| 01_core_types        | 10 min     | Beginner     | Data structures, metadata, traces           |
| 02_supported_formats | 5 min      | Beginner     | File format overview, loader reference      |
| **Total**            | **20 min** | **Beginner** | **Foundational concepts**                   |

---

## Demo Descriptions

### Demo 00: Hello World

**File**: `00_hello_world.py`

**What it teaches**:

- Your Oscura installation works correctly
- Basic API usage pattern: load → measure → analyze
- Common measurements: amplitude, frequency, RMS voltage
- Visualization basics

**What you'll do**:

1. Generate a synthetic 1 kHz sine wave signal
2. Measure its amplitude (peak-to-peak voltage)
3. Measure its frequency
4. Measure its RMS voltage
5. Validate all measurements match expected values

**Expected output**:

```
Amplitude (Vpp): 2.0000 V
Frequency: 1000.00 Hz
RMS voltage: 0.7071 V
```

**Concepts introduced**:

- `WaveformTrace` - The fundamental signal container
- `TraceMetadata` - Signal timing and acquisition information
- Measurement functions (`amplitude()`, `frequency()`, `rms()`)
- Validation patterns

**Next step**: If this demo passes, your installation is working! Move to 01_core_types.

---

### Demo 01: Core Types

**File**: `01_core_types.py`

**What it teaches**:

- Oscura's fundamental data structures
- The difference between analog, digital, and protocol data
- How to create and work with traces
- Metadata and calibration information
- Type conversions and compatibility

**What you'll do**:

1. Create `TraceMetadata` with acquisition parameters
2. Build a `WaveformTrace` from raw signal data
3. Create a `DigitalTrace` from digital/logic signals
4. Construct a `ProtocolPacket` from decoded data
5. Explore properties and relationships between types
6. Access and modify trace attributes

**Key data structures**:

- `TraceMetadata` - Timing info, sample rate, channel names, source file
- `WaveformTrace` - Analog signals (voltage vs time)
- `DigitalTrace` - Digital/logic signals (high/low states)
- `ProtocolPacket` - Decoded protocol data (addresses, values, status)
- `CalibrationInfo` - Instrument calibration details

**Concepts introduced**:

- Type safety and data validation
- Property access patterns
- Metadata interpretation
- IEEE 1241-2010 ADC terminology

**Why this matters**:
Every advanced capability in Oscura works with these core types. Understanding their structure, properties, and relationships is essential for using the framework effectively.

---

### Demo 02: Supported Formats

**File**: `02_supported_formats.py`

**What it teaches**:

- All 21+ file formats Oscura can load
- How file formats are organized (oscilloscopes, logic analyzers, etc.)
- Auto-detection of file format
- Loader selection and usage patterns
- Multi-channel file handling

**What you'll do**:

1. Query all supported file formats
2. View formats organized by equipment category
3. See detailed information for each format
4. Learn typical usage for each category
5. Understand multi-channel and mixed-signal capabilities

**Supported categories**:

| Category           | Formats           | Examples                  |
| ------------------ | ----------------- | ------------------------- |
| Oscilloscopes      | .wfm, .bin, .csv  | Tektronix, Rigol, Siglent |
| Logic Analyzers    | .vcd, .vvp, .jtag | ModelSim, FPGA tools      |
| Automotive         | .dbc, .blf, .asc  | CAN, LIN bus protocols    |
| General Scientific | .csv, .txt, .mat  | MATLAB, spreadsheets      |
| Spectrum Analyzers | .s2p, .bin        | Microwave measurements    |

**Concepts introduced**:

- File format diversity in hardware testing
- Auto-detection and magic bytes
- Loader registry and plugin system
- Industry-standard formats

**Why this matters**:
Real-world reverse engineering involves data from many different instruments. Oscura's format support means you can work with any standard test equipment file format.

---

## How to Run the Demos

### Option 1: Run Individual Demo

Run a single demo to validate a specific concept:

```bash
# From the project root
python demonstrations/00_getting_started/00_hello_world.py

# Or from the demo directory
cd demonstrations/00_getting_started
python 00_hello_world.py
```

Expected output: Green success messages with measurements shown.

### Option 2: Run All Getting Started Demos

Run all three demos in sequence:

```bash
# From the project root
python demonstrations/00_getting_started/00_hello_world.py && \
python demonstrations/00_getting_started/01_core_types.py && \
python demonstrations/00_getting_started/02_supported_formats.py
```

### Option 3: Validate Your Installation

Validate all demonstrations in the project (includes getting started):

```bash
# From the project root
python demonstrations/validate_all.py
```

This runs all demonstrations and reports coverage, validation status, and any failures.

---

## Understanding the Output

When you run a demo, you'll see output like:

```
====================================================================
Oscura Hello World
====================================================================

Welcome to Oscura - the hardware reverse engineering framework!
This demonstration shows the simplest possible workflow.

Signal Information
------------------
Sample rate: 1.00e+05 Hz
Number of samples: 1000
Duration: 0.0100 s

Measurements
------------------
Amplitude (Vpp): 2.0000 V
Frequency: 1000.00 Hz
RMS voltage: 0.7071 V

✓ All measurements validated!
```

**Key elements**:

- Sections with double-line headers (====) organize the output
- Subsections with dashes (---) group related information
- Results show measurements with units
- ✓ indicators show validated results
- Warnings and errors appear in red/yellow

---

## Common Issues and Solutions

### "ModuleNotFoundError: No module named 'oscura'"

**Solution**: Oscura isn't installed. Install it with:

```bash
pip install oscura
# OR with uv
uv add oscura
```

Then verify:

```bash
python -c "import oscura; print(oscura.__version__)"
```

### "Python 3.12+ required"

**Solution**: Oscura requires Python 3.12 or newer. Check your version:

```bash
python --version
```

If you have an older Python, install Python 3.12+ from python.org or use a version manager (pyenv, conda, etc.).

### Demo runs but produces unexpected values

**Solution**: This usually means you're working with different data than expected. The demos generate synthetic data, so results should be exact. If values differ significantly:

1. Check that your Oscura version is current: `pip install --upgrade oscura`
2. Verify Python version: `python --version` (should be 3.12+)
3. Run the validation to see detailed error messages: `python demonstrations/validate_all.py`

### "FileNotFoundError" when loading test data

**Solution**: Some demos generate test data on the fly. Ensure you're running the demo from the correct directory, or generate test data first:

```bash
python demonstrations/generate_all_data.py
```

---

## Next Steps: Where to Go After Getting Started

### If You Want to...

| Goal                                 | Next Demo                                         | Path                                     |
| ------------------------------------ | ------------------------------------------------- | ---------------------------------------- |
| Load and analyze oscilloscope files  | `01_data_loading/01_oscilloscopes.py`             | File I/O → Oscilloscope data             |
| Decode protocols (UART, SPI, I2C)    | `03_protocol_decoding/01_serial_comprehensive.py` | Protocol → Serial decoding               |
| Perform advanced measurements        | `02_basic_analysis/01_waveform_measurements.py`   | Analysis → Measurements                  |
| Reverse engineer an unknown protocol | `06_reverse_engineering/01_unknown_protocol.py`   | Reverse engineering → Protocol inference |
| Analyze power supplies and noise     | `04_advanced_analysis/02_power_analysis.py`       | Advanced → Power analysis                |
| Build a complete RE workflow         | `16_complete_workflows/01_protocol_discovery.py`  | Workflows → Complete examples            |

### Recommended Learning Sequence

1. **Complete Getting Started** (this section)
   - Foundation for all other capabilities
   - Validates your installation
   - Introduces core concepts

2. **Learn Data Loading** (01_data_loading/)
   - How to load real instrument data
   - Working with files from oscilloscopes, logic analyzers
   - Understanding file format diversity

3. **Master Basic Analysis** (02_basic_analysis/)
   - Essential measurements (amplitude, frequency, power)
   - Statistical analysis (mean, std dev, histograms)
   - Spectral analysis (FFT, frequency domain)

4. **Explore Protocol Decoding** (03_protocol_decoding/)
   - Decode standard protocols (UART, SPI, I2C, CAN)
   - Protocol-specific analysis
   - Packet extraction and interpretation

5. **Advanced Topics** (04_advanced_analysis/, 05_domain_specific/, etc.)
   - Complex analysis techniques
   - Industry-specific workflows
   - Reverse engineering methods

---

## Tips for Learning

### Read the Docstrings

Each demo includes extensive documentation at the top:

```python
"""Hello World: Your first Oscura demonstration

Demonstrates:
- oscura.load() - Load signal data
- oscura.amplitude() - Measure peak-to-peak voltage
...
"""
```

Read these first to understand what the demo teaches!

### Modify and Experiment

These demos are meant to be modified. Try:

```python
# Change the frequency
trace = generate_sine_wave(frequency=5000.0)  # 5 kHz instead of 1 kHz

# Add a second signal
trace2 = generate_sine_wave(frequency=2000.0, amplitude=0.5)

# Stack them together
combined = trace.data + trace2.data
```

See what happens! This is the best way to understand the concepts.

### Check the Related Demos

Each demo lists related demonstrations at the top. Use these to see how concepts are applied in different contexts:

```python
Related Demos:
- 00_getting_started/00_hello_world.py
- 02_basic_analysis/01_waveform_measurements.py
```

### Read Real Code

After understanding the basics here, look at:

- `src/oscura/` - The actual framework implementation
- `tests/` - Real test cases showing expected behavior
- `examples/` - Complete working examples

---

## Understanding the Framework

### Core Concepts

**Traces** - The fundamental data structure representing signal measurements:

- `WaveformTrace` - Analog signals (voltage over time)
- `DigitalTrace` - Digital signals (high/low logic levels)

**Metadata** - Information about how the signal was captured:

- Sample rate (Hz) - How often samples were taken
- Vertical scale (V/div) - Voltage per division
- Channel information - Which instrument channel measured this
- Source file - Where the data came from

**Measurements** - Quantities extracted from signals:

- Amplitude - Peak-to-peak voltage
- Frequency - Oscillation rate
- RMS - Root mean square (average power)
- Power - Signal power (watts)
- Etc.

**Protocols** - Decoded communication:

- UART, SPI, I2C, CAN, LIN, Modbus, etc.
- Each has standard packet structure
- Decoded from digital signals

### Architecture Philosophy

Oscura is built on these principles:

1. **Type safety** - Explicit types prevent errors
2. **Metadata everywhere** - Every trace carries context
3. **Measurement accuracy** - Follows IEEE standards
4. **Protocol standards** - Implements industry-standard decoders
5. **Extensibility** - Add custom measurements and decoders
6. **Progressive disclosure** - Simple API for common tasks, advanced API for complex work

---

## Resources

### In This Repository

- **`demonstrations/`** - 100+ working examples
- **`src/oscura/`** - Framework source code
- **`tests/`** - Test cases with expected behavior
- **`docs/`** - Full API documentation

### External Resources

- **[IEEE 1241-2010](https://standards.ieee.org/)** - ADC Terminology (what we measure)
- **[Oscilloscope fundamentals](https://www.tek.com/)** - From Tektronix
- **[Protocol specifications](https://www.iot.org/)** - From IoT organizations

### Getting Help

1. Check the related demos - similar use cases often already exist
2. Read the docstrings in the code - they include examples
3. Look at the test cases for expected behavior
4. Try the `/help` command for framework documentation
5. Examine other demos in the demonstrations/ directory

---

## Summary

The Getting Started section covers:

| Demo                 | Focus                   | Outcome                             |
| -------------------- | ----------------------- | ----------------------------------- |
| 00_hello_world       | Installation validation | Know Oscura works                   |
| 01_core_types        | Data structures         | Understand traces, metadata, types  |
| 02_supported_formats | File format overview    | See what data sources are supported |

After completing these three 20-minute demonstrations, you'll understand:

- How to work with Oscura's core API
- The data structures used throughout the framework
- What file formats are supported
- How to validate your installation
- Where to go for more advanced topics

**Ready to start?** Run this to validate your installation:

```bash
python demonstrations/00_getting_started/00_hello_world.py
```

Happy reverse engineering!
