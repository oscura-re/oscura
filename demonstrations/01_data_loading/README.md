# Data Loading

**Master file I/O and multi-format data ingestion for hardware reverse engineering.**

This section contains 7 demonstrations covering oscilloscope, logic analyzer, automotive, scientific, and custom binary formats. Learn how to load data from any standard test equipment and handle multi-channel captures efficiently.

---

## Prerequisites

Before running these demonstrations, ensure you have:

- **Completed Getting Started** - Finish `00_getting_started/` first
- **Python 3.12+** - Oscura requires Python 3.12 or higher
- **Oscura installed** - Install with `pip install oscura` or `uv add oscura`
- **Basic file format knowledge** - Understanding CSV, binary formats helps but isn't required
- **Optional dependencies** - Some demos require h5py, scipy.io for specific formats

Check optional dependencies:

```bash
pip install oscura[all]  # Includes all optional format loaders
# OR
uv add oscura --all-extras
```

---

## Demonstrations

| Demo      | File                          | Time       | Difficulty               | Topics                                 |
| --------- | ----------------------------- | ---------- | ------------------------ | -------------------------------------- |
| **01**    | `01_oscilloscopes.py`         | 10 min     | Beginner                 | Tektronix, Rigol, LeCroy formats       |
| **02**    | `02_logic_analyzers.py`       | 10 min     | Beginner                 | Sigrok, VCD, Saleae formats            |
| **03**    | `03_automotive_formats.py`    | 10 min     | Intermediate             | Vector BLF/ASC, MDF/MF4, DBC           |
| **04**    | `04_scientific_formats.py`    | 10 min     | Intermediate             | TDMS, HDF5, NPZ, WAV                   |
| **05**    | `05_custom_binary.py`         | 15 min     | Advanced                 | Binary loader API, struct parsing      |
| **06**    | `06_streaming_large_files.py` | 15 min     | Advanced                 | Chunked loading, memory efficiency     |
| **07**    | `07_multi_channel.py`         | 10 min     | Intermediate             | Multi-channel loading, synchronization |
| **Total** |                               | **80 min** | **Beginner to Advanced** | **Complete file I/O mastery**          |

---

## Learning Path

These demonstrations are designed to be completed **in order**. Each builds on concepts from the previous one:

```
01_oscilloscopes.py → 02_logic_analyzers.py → 03_automotive_formats.py
        ↓                       ↓                          ↓
  Analog waveforms       Digital captures         Industry formats
  Metadata extraction    Timing analysis         DBC databases
        ↓                       ↓                          ↓
04_scientific_formats.py → 05_custom_binary.py → 06_streaming_large_files.py
        ↓                       ↓                          ↓
  Multi-format support   Binary parsing          Memory management
  HDF5 hierarchies       Endianness handling     Chunked iterators
        ↓
07_multi_channel.py
        ↓
  Synchronized analysis
  Cross-channel correlation
```

### Recommended Time

**Beginner path** (40 min): Demos 01, 02, 07
**Intermediate path** (60 min): Demos 01-04, 07
**Advanced path** (80 min): All demos

---

## Key Concepts

### What You'll Learn

**Oscilloscope Formats** (Demo 01):

- Tektronix .wfm file structure and metadata
- Rigol and LeCroy proprietary formats
- Vertical scale, coupling, and probe settings
- Multi-format detection and auto-loading

**Logic Analyzer Formats** (Demo 02):

- Sigrok .sr archive structure
- VCD (Value Change Dump) from simulators
- Saleae binary capture format
- Digital channel timing extraction

**Automotive Formats** (Demo 03):

- Vector BLF (Binary Logging Format)
- Vector ASC (ASCII log files)
- ASAM MDF/MF4 measurement data
- DBC database signal definitions
- CAN, CAN-FD, LIN, FlexRay handling

**Scientific Formats** (Demo 04):

- TDMS (LabVIEW/National Instruments)
- HDF5 hierarchical datasets
- NumPy compressed arrays (NPZ)
- WAV audio as waveform data

**Custom Binary** (Demo 05):

- Binary loader API usage
- Custom header parsing with struct
- Endianness handling (big/little endian)
- Multi-channel interleaved data

**Streaming Large Files** (Demo 06):

- Chunked loading to avoid memory exhaustion
- Lazy loading with deferred access
- Progress tracking for long operations
- Memory usage monitoring

**Multi-Channel** (Demo 07):

- load_all_channels() API
- Channel selection and indexing
- Synchronized multi-trace analysis
- Cross-channel correlation

---

## Running Demonstrations

### Option 1: Run Individual Demo

Run a single demo to focus on a specific format:

```bash
# From the project root
python demonstrations/01_data_loading/01_oscilloscopes.py

# Or from the demo directory
cd demonstrations/01_data_loading
python 01_oscilloscopes.py
```

Expected output: Format-specific metadata and successful validation.

### Option 2: Run All Data Loading Demos

Run all seven demos in sequence:

```bash
# From the project root
for demo in demonstrations/01_data_loading/*.py; do
    python "$demo"
done
```

### Option 3: Validate All Demonstrations

Validate all demonstrations in the project:

```bash
# From the project root
python demonstrations/validate_all.py
```

This runs all demonstrations and reports coverage, validation status, and failures.

---

## What You'll Learn

### File Format Categories

**Test Equipment Formats**:

- Oscilloscopes: .wfm, .bin, .trc, .csv
- Logic Analyzers: .sr, .vcd, .sal
- Spectrum Analyzers: .s2p

**Industry Standards**:

- Automotive: .blf, .asc, .dbc, .mf4
- Scientific: .tdms, .hdf5, .npz
- Audio: .wav, .flac

**Custom Formats**:

- Binary loader API for proprietary formats
- Header parsing and validation
- Multi-channel data extraction

### Advanced Techniques

**Format Detection**:

- Magic byte identification
- Header structure analysis
- Auto-detection and fallback strategies

**Metadata Extraction**:

- Sample rate and timing information
- Vertical scale and calibration
- Channel names and acquisition settings
- Source file and instrument details

**Memory Management**:

- Chunked loading for files exceeding RAM
- Streaming iterators for continuous processing
- Lazy loading for deferred access
- Memory usage monitoring and bounds checking

**Multi-Channel Handling**:

- Simultaneous channel loading
- Channel alignment validation
- Synchronized analysis across traces
- Cross-channel relationships (correlation, phase)

---

## Common Issues and Solutions

### "ModuleNotFoundError: No module named 'h5py'"

**Solution**: Some demos require optional dependencies. Install them:

```bash
pip install oscura[all]
# OR
pip install h5py scipy
```

### "FileNotFoundError" when loading test data

**Solution**: Most demos generate synthetic test data on the fly. If you see this error:

1. Ensure you're running from the correct directory
2. Check that Python can write temporary files
3. Run with elevated permissions if needed

### "UnsupportedFormatError: Format not recognized"

**Solution**: The file format may not be supported or the file is corrupted:

1. Check the file extension matches the format
2. Verify the file isn't corrupted (check size, open in hex editor)
3. Consult `00_getting_started/02_supported_formats.py` for supported formats
4. For custom formats, use `05_custom_binary.py` as a template

### Memory errors when loading large files

**Solution**: Use streaming and chunked loading:

1. Run `06_streaming_large_files.py` to learn techniques
2. Use chunked iterators instead of loading entire file
3. Monitor memory usage with the techniques shown
4. Consider processing in segments

### Multi-channel alignment issues

**Solution**: Validate timing synchronization:

1. Check that all channels have same sample rate
2. Verify trigger alignment in acquisition settings
3. Use techniques from `07_multi_channel.py` for validation
4. Look for clock skew in metadata

---

## Next Steps: Where to Go After Data Loading

### If You Want to...

| Goal                              | Next Demo                                         | Path                             |
| --------------------------------- | ------------------------------------------------- | -------------------------------- |
| Analyze loaded waveforms          | `02_basic_analysis/01_waveform_measurements.py`   | File I/O → Measurements          |
| Decode protocols from loaded data | `03_protocol_decoding/01_serial_comprehensive.py` | File I/O → Protocol decoding     |
| Perform statistical analysis      | `02_basic_analysis/02_statistics.py`              | File I/O → Statistics            |
| Analyze automotive captures       | `05_domain_specific/01_automotive_diagnostics.py` | Automotive loading → Diagnostics |
| Work with frequency domain        | `02_basic_analysis/03_spectral_analysis.py`       | File I/O → FFT/PSD               |

### Recommended Learning Sequence

1. **Master Data Loading** (this section)
   - Load data from any instrument
   - Handle multi-format captures
   - Manage memory efficiently

2. **Learn Basic Analysis** (02_basic_analysis/)
   - Apply measurements to loaded data
   - Statistical characterization
   - Frequency domain analysis

3. **Explore Protocol Decoding** (03_protocol_decoding/)
   - Decode protocols from loaded captures
   - Extract communication packets
   - Validate protocol compliance

4. **Advanced Topics** (04_advanced_analysis/, 05_domain_specific/)
   - Domain-specific workflows
   - Advanced signal processing
   - Industry-standard compliance

---

## Tips for Learning

### Start Simple

Begin with oscilloscope and logic analyzer formats before tackling automotive or custom binary formats. The concepts build progressively.

### Generate Test Data

Most demos generate synthetic data, but you can also:

```python
# Generate test data for experimentation
python demonstrations/generate_all_data.py

# Then load your own files
from oscura import load
trace = load("path/to/your/file.wfm")
```

### Examine Metadata

Every loaded trace carries rich metadata:

```python
trace = load("capture.wfm")
print(f"Sample rate: {trace.metadata.sample_rate} Hz")
print(f"Duration: {len(trace.data) / trace.metadata.sample_rate} s")
print(f"Source: {trace.metadata.source_file}")
```

Understanding metadata is key to correct analysis.

### Try Different Formats

Load the same signal in different formats to understand format-specific features:

```python
# Load oscilloscope capture
osc_trace = load("signal.wfm")

# Export and reload as CSV
export_csv(osc_trace, "signal.csv")
csv_trace = load("signal.csv")

# Compare metadata
compare_metadata(osc_trace, csv_trace)
```

### Handle Errors Gracefully

Real-world files are messy. Learn error handling:

```python
try:
    trace = load("capture.bin")
except UnsupportedFormatError:
    print("Format not recognized - use binary loader API")
    trace = load_binary("capture.bin", custom_parser)
```

---

## Understanding the Framework

### Core Loading API

**Simple Loading**:

```python
from oscura import load

# Auto-detect format and load
trace = load("capture.wfm")
```

**Multi-Channel Loading**:

```python
from oscura import load_all_channels

# Load all channels simultaneously
traces = load_all_channels("multi_channel.wfm")
ch1 = traces["CH1"]
ch2 = traces["CH2"]
```

**Streaming Loading**:

```python
from oscura.loaders import load_chunked

# Load in chunks to avoid memory exhaustion
for chunk in load_chunked("large_file.bin", chunk_size=10000):
    process(chunk)
```

### Format Registry

Oscura maintains a registry of supported formats:

```python
from oscura.loaders import get_supported_formats

formats = get_supported_formats()
for fmt in formats:
    print(f"{fmt.extension}: {fmt.description}")
```

### Custom Format Support

For proprietary formats:

```python
from oscura.loaders.binary import load_binary

def parse_header(data):
    # Custom header parsing
    return metadata

trace = load_binary("proprietary.bin", header_parser=parse_header)
```

---

## Resources

### In This Repository

- **`src/oscura/loaders/`** - Loader implementations
- **`tests/integration/loaders/`** - Loader test cases
- **`examples/formats/`** - Real-world format examples

### External Resources

- **[Sigrok File Formats](https://sigrok.org/)** - Logic analyzer formats
- **[Vector Tools](https://www.vector.com/)** - Automotive formats
- **[HDF5 Documentation](https://www.hdfgroup.org/)** - HDF5 hierarchical data
- **[LabVIEW TDMS](https://www.ni.com/)** - National Instruments formats

### IEEE Standards

- **IEEE 181-2011** - Waveform and vector measurements
- **IEEE 1057-2017** - Digitizing waveform recorders
- **IEEE 1364-2005** - Verilog VCD format

### Getting Help

1. Check demo docstrings for detailed examples
2. Examine loader source in `src/oscura/loaders/`
3. Review test cases for expected behavior
4. Try the `/help` command for format documentation
5. Consult supported formats in `00_getting_started/02_supported_formats.py`

---

## Summary

The Data Loading section covers:

| Demo                     | Focus                   | Outcome                             |
| ------------------------ | ----------------------- | ----------------------------------- |
| 01_oscilloscopes         | Analog waveform formats | Load Tektronix, Rigol, LeCroy files |
| 02_logic_analyzers       | Digital capture formats | Load Sigrok, VCD, Saleae files      |
| 03_automotive_formats    | Automotive protocols    | Load BLF, ASC, MDF, DBC files       |
| 04_scientific_formats    | Scientific data         | Load TDMS, HDF5, NPZ, WAV files     |
| 05_custom_binary         | Binary parsing          | Create custom format loaders        |
| 06_streaming_large_files | Memory management       | Handle files exceeding RAM          |
| 07_multi_channel         | Multi-trace analysis    | Synchronized channel loading        |

After completing these seven 80-minute demonstrations, you'll understand:

- How to load data from 21+ file formats
- Format detection and auto-loading strategies
- Metadata extraction and validation
- Memory-efficient streaming for large files
- Multi-channel synchronized analysis
- Custom binary format parsing

**Ready to start?** Run this to begin with oscilloscope formats:

```bash
python demonstrations/01_data_loading/01_oscilloscopes.py
```

Happy signal loading!
