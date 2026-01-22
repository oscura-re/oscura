# Error Code Reference

> **Version**: 0.5.0 | **Last Updated**: 2026-01-19 | **Applies to**: Oscura 0.5.0

This document provides a complete reference for all error codes used by Oscura.

## Error Code Format

All error codes follow the format: `TK-<CATEGORY>-<NUMBER>`

- **TK**: Oscura prefix
- **CATEGORY**: Three-letter category code
- **NUMBER**: Three-digit error number within category

## Error Categories

|Category|Code Range|Description|
|---|---|---|
|GEN|001-099|General/uncategorized errors|
|FILE|100-199|File system and I/O errors|
|FMT|200-299|File format errors|
|LOAD|300-399|Waveform loading errors|
|MEAS|400-499|Measurement errors|
|DEC|500-599|Protocol decoding errors|
|INF|600-699|Protocol inference errors|
|FFT|700-799|Spectral analysis errors|
|RPT|800-899|Report generation errors|
|CFG|900-999|Configuration errors|

---

## General Errors (TK-GEN-xxx)

### TK-GEN-001: Unknown Error

**Severity:** Error

An unexpected error occurred that doesn't fit into other categories.

**Common Causes:**

- Unhandled exception in code
- System-level errors

**Resolution:**

- Check the full error traceback for details
- Report the issue with complete error output

---

### TK-GEN-002: Operation Cancelled

**Severity:** Info

The operation was cancelled by the user.

**Common Causes:**

- User pressed Ctrl+C
- Timeout exceeded

**Resolution:**

- Re-run the command if needed

---

### TK-GEN-003: Not Implemented

**Severity:** Error

The requested feature is not yet implemented.

**Common Causes:**

- Using a feature from documentation for a future version
- Platform-specific feature not available

**Resolution:**

- Check the version compatibility
- Consider contributing the feature

---

## File Errors (TK-FILE-xxx)

### TK-FILE-101: File Not Found

**Severity:** Error

The specified file could not be located.

**Common Causes:**

- Incorrect file path
- File was moved or deleted
- Typo in filename

**Resolution:**

- Verify the file path is correct
- Use absolute paths to avoid confusion
- Check file exists: `ls -la /path/to/file`

**Example:**

```
Error [TK-FILE-101]: File not found: /path/to/capture.wfm

  The specified waveform file could not be located.

Suggestion: Check that the file path is correct and the file exists.
```

---

### TK-FILE-102: Permission Denied

**Severity:** Error

Insufficient permissions to access the file.

**Common Causes:**

- File owned by another user
- Read-only file system
- Restrictive permissions

**Resolution:**

- Check file permissions: `ls -la /path/to/file`
- Change ownership or permissions if appropriate
- Run with appropriate user privileges

---

### TK-FILE-103: File Already Exists

**Severity:** Warning

The output file already exists and would be overwritten.

**Common Causes:**

- Re-running analysis with same output path
- Export to existing file

**Resolution:**

- Use `--force` flag to overwrite
- Choose a different output filename
- Backup existing file first

---

### TK-FILE-104: Disk Full

**Severity:** Error

Insufficient disk space for the operation.

**Common Causes:**

- Large export file
- Many temporary files
- Full disk

**Resolution:**

- Check disk space: `df -h`
- Clear temporary files
- Use a different output location

---

### TK-FILE-105: Write Error

**Severity:** Error

Could not write to the file.

**Common Causes:**

- Permission issues
- Disk full
- File locked by another process

**Resolution:**

- Check write permissions
- Verify disk space available
- Close other applications using the file

---

## Format Errors (TK-FMT-xxx)

### TK-FMT-201: Unknown Format

**Severity:** Error

The file format could not be determined.

**Common Causes:**

- Unrecognized file extension
- Missing magic bytes
- Corrupted file header

**Resolution:**

- Specify format explicitly: `osc.load("file", format="tektronix")`
- Check supported formats: `osc.get_supported_formats()`
- Verify file is not corrupted

**Example:**

```
Error [TK-FMT-201]: Cannot determine file format: data.bin

  Supported formats: .wfm, .csv, .npz, .h5, .sr, .trc

Suggestion: Specify format explicitly using format= parameter.
```

---

### TK-FMT-202: Unsupported Format

**Severity:** Error

The file format is recognized but not supported.

**Common Causes:**

- Proprietary format
- Newer format version
- Platform-specific format

**Resolution:**

- Convert file to supported format
- Check for format-specific plugins
- Export from source application in different format

---

### TK-FMT-203: Corrupt File

**Severity:** Error

The file appears to be corrupted or incomplete.

**Common Causes:**

- Interrupted transfer
- Storage corruption
- Incomplete capture

**Resolution:**

- Re-download or re-capture the file
- Check file integrity with checksums
- Try recovery tools if available

---

### TK-FMT-204: Version Mismatch

**Severity:** Error

The file format version is not compatible.

**Common Causes:**

- File from newer instrument firmware
- File from older unsupported version

**Resolution:**

- Update Oscura to latest version
- Re-export from instrument with compatible settings
- Check format documentation

---

## Loading Errors (TK-LOAD-xxx)

### TK-LOAD-301: No Waveform Data

**Severity:** Error

The file contains no waveform data.

**Common Causes:**

- Empty capture file
- File contains only metadata
- Wrong dataset selected (HDF5)

**Resolution:**

- Verify capture was successful
- Check file contains data: `osc.info("file.wfm")`
- For HDF5, specify correct dataset path

---

### TK-LOAD-302: Invalid Sample Rate

**Severity:** Error

The sample rate in the file is invalid or missing.

**Common Causes:**

- Corrupted metadata
- Manual file editing
- Conversion error

**Resolution:**

- Specify sample rate manually: `osc.load("file", sample_rate=1e9)`
- Re-capture or re-export file

---

### TK-LOAD-303: Channel Not Found

**Severity:** Error

The requested channel does not exist in the file.

**Common Causes:**

- Wrong channel name
- Channel not captured
- Multi-file capture

**Resolution:**

- List available channels: `osc.get_channels("file.wfm")`
- Use correct channel name
- Check if channel is in separate file

**Example:**

```
Error [TK-LOAD-303]: Channel 'CH5' not found

  Available channels: CH1, CH2, CH3, CH4, D0-D15

Suggestion: Use one of the available channel names.
```

---

### TK-LOAD-304: Memory Limit Exceeded

**Severity:** Error

The file is too large to load into memory.

**Common Causes:**

- Very large capture file
- Insufficient system RAM
- Multiple large files loaded

**Resolution:**

- Use lazy loading: `osc.load("file", lazy=True)`
- Process in chunks: `osc.iter_chunks()`
- Increase system memory or swap

---

### TK-LOAD-305: Decompression Error

**Severity:** Error

Failed to decompress file data.

**Common Causes:**

- Corrupted compressed data
- Unsupported compression algorithm
- Truncated file

**Resolution:**

- Verify file integrity
- Re-download or re-capture
- Try uncompressed export

---

## Measurement Errors (TK-MEAS-xxx)

### TK-MEAS-401: Insufficient Data

**Severity:** Warning

Not enough data points for the measurement.

**Common Causes:**

- Very short capture
- Measurement requires more cycles
- Decimation too aggressive

**Resolution:**

- Use longer capture
- Adjust measurement parameters
- Reduce decimation

---

### TK-MEAS-402: No Transitions

**Severity:** Warning

Signal has no edge transitions for timing measurement.

**Common Causes:**

- DC signal
- Signal below/above threshold
- Incorrect threshold setting

**Resolution:**

- Verify signal has transitions
- Adjust threshold: `osc.measure_frequency(trace, threshold=0.5)`
- Check signal is not stuck high/low

**Example:**

```
Warning [TK-MEAS-402]: No transitions detected in signal

  Threshold: 1.65 V
  Signal range: 0.02 V to 0.05 V

Suggestion: Adjust threshold or verify signal contains edges.
```

---

### TK-MEAS-403: Measurement Not Applicable

**Severity:** Warning

The measurement is not applicable to this signal type.

**Common Causes:**

- Rise time on non-pulse signal
- Frequency on aperiodic signal
- Digital measurement on analog signal

**Resolution:**

- Use appropriate measurement for signal type
- Verify signal characteristics first

---

### TK-MEAS-404: NaN Result

**Severity:** Warning

The measurement returned NaN (Not a Number).

**Common Causes:**

- Signal doesn't meet measurement criteria
- Division by zero in calculation
- Invalid input data

**Resolution:**

- Check signal meets measurement requirements
- Verify data is valid (no NaN in input)

---

### TK-MEAS-405: Out of Range

**Severity:** Warning

The measurement result is outside expected range.

**Common Causes:**

- Noise affecting measurement
- Incorrect units/scaling
- Algorithm limitation

**Resolution:**

- Check signal quality
- Verify input parameters
- Use alternative measurement method

---

## Decode Errors (TK-DEC-xxx)

### TK-DEC-501: Invalid Protocol Configuration

**Severity:** Error

The protocol decoder configuration is invalid.

**Common Causes:**

- Invalid baud rate
- Incompatible parameter combination
- Missing required parameter

**Resolution:**

- Check protocol documentation
- Verify all required parameters
- Use protocol inference if unsure

---

### TK-DEC-502: Sync Lost

**Severity:** Warning

Protocol synchronization was lost during decoding.

**Common Causes:**

- Signal corruption
- Baud rate mismatch
- Noise on signal

**Resolution:**

- Verify protocol parameters
- Check signal quality
- Use auto-baud detection

---

### TK-DEC-503: Framing Error

**Severity:** Warning

Protocol framing error detected.

**Common Causes:**

- Wrong number of data/stop bits
- Parity mismatch
- Noise corruption

**Resolution:**

- Verify protocol settings match transmitter
- Check for noise on signal
- Examine waveform at error location

**Example:**

```
Warning [TK-DEC-503]: UART framing error at 1.234 ms

  Expected stop bit (high), found low
  Baud rate: 115200

Suggestion: Verify stop bit configuration or check for noise.
```

---

### TK-DEC-504: Parity Error

**Severity:** Warning

Parity check failed on decoded byte.

**Common Causes:**

- Wrong parity setting
- Bit error in transmission
- Noise corruption

**Resolution:**

- Verify parity setting matches transmitter
- Check signal integrity
- Note: some errors may be in original data

---

### TK-DEC-505: CRC Error

**Severity:** Warning

CRC/checksum verification failed.

**Common Causes:**

- Data corruption
- Wrong CRC polynomial
- Partial message capture

**Resolution:**

- Verify CRC configuration
- Check for complete message capture
- May indicate actual transmission error

---

### TK-DEC-506: Timeout

**Severity:** Warning

Protocol decoder timed out waiting for data.

**Common Causes:**

- Incomplete transaction captured
- Very slow baud rate
- Clock/data synchronization issue

**Resolution:**

- Extend capture duration
- Check clock configuration
- Verify all signal connections

---

## Inference Errors (TK-INF-xxx)

### TK-INF-601: No Protocol Detected

**Severity:** Warning

Could not automatically detect protocol type.

**Common Causes:**

- Unknown protocol
- Signal too noisy
- Insufficient data

**Resolution:**

- Manually specify protocol type
- Improve signal quality
- Capture more data

---

### TK-INF-602: Low Confidence

**Severity:** Warning

Protocol detected but with low confidence.

**Common Causes:**

- Ambiguous signal characteristics
- Multiple possible protocols
- Noisy signal

**Resolution:**

- Review suggested protocols
- Verify with known data if possible
- Try different confidence threshold

---

### TK-INF-603: Parameter Estimation Failed

**Severity:** Warning

Could not estimate protocol parameters.

**Common Causes:**

- Non-standard parameters
- Insufficient data
- Signal quality issues

**Resolution:**

- Try with manual parameter hints
- Capture more data
- Improve signal quality

---

## FFT Errors (TK-FFT-xxx)

### TK-FFT-701: Invalid FFT Size

**Severity:** Error

The specified FFT size is invalid.

**Common Causes:**

- Non-power-of-two size
- Size larger than data
- Zero or negative size

**Resolution:**

- Use power-of-two FFT size
- Ensure data is long enough
- Use default size if unsure

---

### TK-FFT-702: Window Error

**Severity:** Error

Invalid window function specified.

**Common Causes:**

- Unknown window name
- Invalid window parameters

**Resolution:**

- Use standard window: hanning, hamming, blackman, etc.
- Check window parameter documentation

---

### TK-FFT-703: GPU Unavailable

**Severity:** Warning

GPU acceleration requested but not available.

**Common Causes:**

- No CUDA GPU present
- CUDA drivers not installed
- cupy not installed

**Resolution:**

- Install CUDA and cupy for GPU support
- Falling back to CPU (slower but functional)
- Check GPU setup guide

---

## Report Errors (TK-RPT-xxx)

### TK-RPT-801: Template Not Found

**Severity:** Error

Report template not found.

**Common Causes:**

- Invalid template name
- Missing template file

**Resolution:**

- Use built-in template: standard, summary, detailed
- Check template path if custom

---

### TK-RPT-802: Render Error

**Severity:** Error

Failed to render report content.

**Common Causes:**

- Missing required data
- Invalid template syntax
- Resource unavailable (fonts, images)

**Resolution:**

- Verify all required data is provided
- Check template for errors
- Ensure resources are accessible

---

### TK-RPT-803: Export Format Error

**Severity:** Error

Cannot export to requested format.

**Common Causes:**

- Unsupported format
- Missing dependencies (reportlab, etc.)

**Resolution:**

- Use supported format: pdf, html, pptx, md
- Install optional dependencies for format

---

## Configuration Errors (TK-CFG-xxx)

### TK-CFG-901: Invalid Configuration

**Severity:** Error

Configuration value is invalid.

**Common Causes:**

- Type mismatch
- Out of range value
- Unknown option

**Resolution:**

- Check configuration documentation
- Use default values if unsure
- Validate config file syntax

---

### TK-CFG-902: Configuration Not Found

**Severity:** Warning

Configuration file not found.

**Common Causes:**

- First run (no config created)
- Config file deleted
- Wrong path

**Resolution:**

- Create config: `oscura config --init`
- Specify path: `--config /path/to/config.yaml`
- Using defaults is usually fine

---

## Getting Help

If you encounter an error not listed here or need additional assistance:

1. **Check documentation**: Review relevant guide or API documentation
2. **Search issues**: Check GitHub issues for similar problems
3. **Debug mode**: Run with `--verbose` or `-vvv` for more details
4. **Report issue**: Include full error output with `--debug` flag

### Debug Mode

For detailed error information:

```bash
# CLI
oscura --verbose --verbose analyze capture.wfm

# Python
import logging
logging.basicConfig(level=logging.DEBUG)

import oscura as osc
trace = osc.load("capture.wfm")
```

This will include:

- Full stack trace
- System information
- Configuration details
- Intermediate values
