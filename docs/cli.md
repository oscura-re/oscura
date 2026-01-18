# CLI Reference

> **Version**: 0.1.0 | **Last Updated**: 2026-01-08 | **Applies to**: Oscura 0.1.0

Command-line interface documentation for Oscura.

## Usage

```bash
oscura [OPTIONS] COMMAND [ARGS]...
```

Or with uv:

```bash
uv run oscura [OPTIONS] COMMAND [ARGS]...
```

## Global Options

|Option|Description|
|---|---|
|`-V, --version`|Show version and exit|
|`--config FILE`|Path to configuration file|
|`--verbose, -v`|Increase verbosity (use multiple times for more)|
|`--quiet, -q`|Suppress non-error output|
|`--no-color`|Disable colored output|
|`--help`|Show help message|

---

## Commands

### analyze

Analyze a waveform file and display measurements.

```bash
oscura analyze FILE [OPTIONS]
```

**Arguments:**

|Argument|Description|
|---|---|
|`FILE`|Waveform file to analyze|

**Options:**

|Option|Description|Default|
|---|---|---|
|`--format, -f`|Output format (text, json, csv)|text|
|`--output, -o`|Output file (default: stdout)|-|
|`--measurements, -m`|Measurements to perform (comma-separated)|all|
|`--channel, -c`|Channel to analyze|auto|

**Available Measurements:**

- `frequency` - Signal frequency
- `period` - Signal period
- `amplitude` - Peak-to-peak amplitude
- `rise_time` - 10-90% rise time
- `fall_time` - 90-10% fall time
- `duty_cycle` - Duty cycle percentage
- `rms` - RMS voltage
- `mean` - Mean voltage
- `thd` - Total harmonic distortion
- `snr` - Signal-to-noise ratio

**Examples:**

```bash
# Basic analysis
oscura analyze capture.wfm

# JSON output
oscura analyze capture.wfm --format json

# Specific measurements
oscura analyze capture.wfm -m frequency,rise_time,duty_cycle

# Save to file
oscura analyze capture.wfm -o results.json --format json
```

**Example Output:**

```
Waveform Analysis: capture.wfm
============================================================
File:          capture.wfm
Samples:       1,000,000
Sample Rate:   1.00 GS/s
Duration:      1.000 ms
Channel:       CH1

Measurements:
  Frequency:     10.00 MHz
  Period:        100.00 ns
  Amplitude:     3.30 V (pk-pk)
  Rise Time:     2.50 ns
  Fall Time:     2.80 ns
  Duty Cycle:    50.2%
  RMS:           1.17 V
  Mean:          0.02 V
```

---

### decode

Decode a protocol from a waveform.

```bash
oscura decode FILE --protocol PROTOCOL [OPTIONS]
```

**Arguments:**

|Argument|Description|
|---|---|
|`FILE`|Waveform file to decode|

**Required Options:**

|Option|Description|
|---|---|
|`--protocol, -p`|Protocol type (uart, spi, i2c, can, etc.)|

**Protocol-Specific Options:**

**UART:**

|Option|Description|Default|
|---|---|---|
|`--baud`|Baud rate|115200|
|`--data-bits`|Data bits (5-9)|8|
|`--parity`|Parity (none, even, odd)|none|
|`--stop-bits`|Stop bits (1, 1.5, 2)|1|

**SPI:**

|Option|Description|Default|
|---|---|---|
|`--clock-channel`|Clock signal channel|-|
|`--data-channel`|Data signal channel|-|
|`--cs-channel`|Chip select channel|-|
|`--cpol`|Clock polarity (0, 1)|0|
|`--cpha`|Clock phase (0, 1)|0|

**I2C:**

|Option|Description|Default|
|---|---|---|
|`--sda-channel`|SDA channel|-|
|`--scl-channel`|SCL channel|-|
|`--address-bits`|Address bits (7, 10)|7|

**CAN:**

|Option|Description|Default|
|---|---|---|
|`--bitrate`|CAN bitrate|500000|
|`--extended`|Extended (29-bit) IDs|false|

**General Options:**

|Option|Description|Default|
|---|---|---|
|`--format, -f`|Output format (text, json, hex, binary)|text|
|`--output, -o`|Output file|stdout|
|`--timestamps`|Include timestamps|true|
|`--errors`|Show decode errors|true|

**Examples:**

```bash
# Decode UART
oscura decode capture.wfm -p uart --baud 9600

# Decode SPI with custom settings
oscura decode multi.wfm -p spi --clock-channel D0 --data-channel D1 --cpol 1 --cpha 1

# Decode I2C and save as JSON
oscura decode capture.wfm -p i2c --sda-channel D0 --scl-channel D1 -f json -o decoded.json

# Decode CAN with extended IDs
oscura decode can_capture.wfm -p can --bitrate 500000 --extended
```

**Example Output:**

```
Protocol Decode: UART @ 115200 baud
============================================================
[0.000000s] TX: 48 65 6C 6C 6F  "Hello"
[0.000434s] TX: 0D 0A           "\r\n"
[0.001205s] RX: 4F 4B           "OK"
[0.001639s] RX: 0D 0A           "\r\n"

Summary:
  Messages decoded: 4
  Bytes transferred: 9
  Errors: 0
```

---

### report

Generate a comprehensive analysis report.

```bash
oscura report FILE [OPTIONS]
```

**Arguments:**

|Argument|Description|
|---|---|
|`FILE`|Waveform file to analyze|

**Options:**

|Option|Description|Default|
|---|---|---|
|`--output, -o`|Output file path|report.pdf|
|`--format, -f`|Output format (pdf, html, pptx, md)|pdf|
|`--title`|Report title|"Signal Analysis Report"|
|`--author`|Report author|-|
|`--template`|Report template|standard|
|`--include-plots`|Include waveform plots|true|
|`--include-raw-data`|Include raw measurement data|false|

**Available Templates:**

- `standard` - Comprehensive technical report
- `summary` - Executive summary only
- `detailed` - Full analysis with all measurements
- `minimal` - Basic measurements only

**Examples:**

```bash
# Generate PDF report
oscura report capture.wfm -o analysis.pdf

# Generate HTML report
oscura report capture.wfm -f html -o report.html

# PowerPoint presentation
oscura report capture.wfm -f pptx -o presentation.pptx --title "Q4 Signal Analysis"

# Markdown for documentation
oscura report capture.wfm -f md -o report.md
```

---

### infer

Automatically detect and infer protocol parameters.

```bash
oscura infer FILE [OPTIONS]
```

**Arguments:**

|Argument|Description|
|---|---|
|`FILE`|Waveform file to analyze|

**Options:**

|Option|Description|Default|
|---|---|---|
|`--format, -f`|Output format (text, json)|text|
|`--confidence`|Minimum confidence threshold|0.7|
|`--max-protocols`|Maximum protocols to suggest|3|

**Examples:**

```bash
# Infer protocol
oscura infer unknown_capture.wfm

# JSON output for scripting
oscura infer capture.wfm -f json
```

**Example Output:**

```
Protocol Inference: unknown_capture.wfm
============================================================

Detected Protocols (confidence >= 70%):

1. UART (confidence: 95.2%)
   - Baud rate: 115200 (estimated)
   - Data bits: 8
   - Parity: none
   - Stop bits: 1

2. SPI (confidence: 23.1%)
   - Not recommended (low confidence)

Recommendation: Use UART decoder with --baud 115200
```

---

### info

Display information about a waveform file.

```bash
oscura info FILE [OPTIONS]
```

**Arguments:**

|Argument|Description|
|---|---|
|`FILE`|Waveform file to inspect|

**Options:**

|Option|Description|Default|
|---|---|---|
|`--format, -f`|Output format (text, json)|text|
|`--verbose, -v`|Show detailed information|false|

**Examples:**

```bash
# Quick info
oscura info capture.wfm

# Detailed info
oscura info capture.wfm -v

# JSON output
oscura info capture.wfm -f json
```

**Example Output:**

```
File Information: capture.wfm
============================================================
Format:        Tektronix WFM
File Size:     15.2 MB
Channels:      4 (CH1, CH2, CH3, CH4)
Sample Rate:   1.00 GS/s
Samples:       10,000,000
Duration:      10.000 ms
Vertical:      8-bit resolution
Created:       2026-01-05 14:32:18
Instrument:    MSO64
```

---

### convert

Convert waveform files between formats.

```bash
oscura convert INPUT OUTPUT [OPTIONS]
```

**Arguments:**

|Argument|Description|
|---|---|
|`INPUT`|Source waveform file|
|`OUTPUT`|Destination file|

**Options:**

|Option|Description|Default|
|---|---|---|
|`--format, -f`|Output format (auto-detected from extension)|auto|
|`--channel, -c`|Channel(s) to export|all|
|`--downsample`|Downsample factor|1|
|`--compress`|Compression level (0-9)|0|

**Supported Output Formats:**

- `.csv` - Comma-separated values
- `.npz` - NumPy compressed archive
- `.h5` / `.hdf5` - HDF5 format
- `.mat` - MATLAB format
- `.json` - JSON with base64 data

**Examples:**

```bash
# Convert to CSV
oscura convert capture.wfm data.csv

# Convert to HDF5 with compression
oscura convert capture.wfm data.h5 --compress 6

# Export specific channel
oscura convert capture.wfm ch1_only.csv --channel CH1

# Downsample for smaller files
oscura convert large.wfm smaller.npz --downsample 10
```

---

### config

Manage Oscura configuration.

```bash
oscura config [OPTIONS]
```

**Options:**

|Option|Description|
|---|---|
|`--init`|Create default configuration file|
|`--show`|Display current configuration|
|`--path`|Show configuration file path|
|`--set KEY=VALUE`|Set a configuration value|

**Examples:**

```bash
# Initialize config
oscura config --init

# Show current settings
oscura config --show

# Set a value
oscura config --set default_format=json
```

---

### version

Display version and system information.

```bash
oscura version [OPTIONS]
```

**Options:**

|Option|Description|
|---|---|
|`--verbose, -v`|Show detailed system info|

**Example Output:**

```
Oscura 0.1.0

Python:     3.12.1
Platform:   Linux-6.8.0-88-generic-x86_64
NumPy:      1.26.3
GPU:        Not available
```

---

## Exit Codes

|Code|Description|
|---|---|
|0|Success|
|1|General error|
|2|Invalid arguments|
|3|File not found|
|4|Unsupported format|
|5|Decode error|
|130|Interrupted (Ctrl+C)|

## Environment Variables

|Variable|Description|
|---|---|
|`OSCURA_CONFIG`|Configuration file path|
|`OSCURA_DATA_DIR`|Default data directory|
|`OSCURA_GPU`|Enable GPU (true/false)|
|`OSCURA_LOG_LEVEL`|Log level (DEBUG, INFO, WARNING, ERROR)|
|`NO_COLOR`|Disable colored output|

## Configuration File

Default location: `~/.config/oscura/config.yaml`

```yaml
# Oscura Configuration
default_sample_rate: 1e9
output_format: text
plot_style: default
gpu_enabled: false

# Protocol defaults
uart:
  baud_rate: 115200
  data_bits: 8
  parity: none

# Report settings
report:
  author: 'Engineering Team'
  template: standard
  include_plots: true
```

## See Also

- [API Reference](api/index.md) - Python API documentation
