# Loader API Reference

> **Version**: 0.9.0
> **Last Updated**: 2026-01-25

## Overview

Oscura provides a unified data loading interface that supports multiple oscilloscope and logic analyzer file formats. The loader system auto-detects file formats and provides consistent trace objects regardless of source.

## Quick Start

```python
import oscura as osc

# Auto-detect format and load
trace = osc.load("capture.wfm")

# Load with explicit format
trace = osc.load("data.bin", format="tektronix")

# Load all channels from multi-channel file
channels = osc.load_all_channels("multi_channel.wfm")

# Lazy loading for large files
trace = osc.load("huge_file.wfm", lazy=True)
```

## Core Functions

### `load()`

Load trace data from file with automatic format detection.

```python
def load(
    path: str | PathLike[str],
    *,
    format: str | None = None,
    channel: str | int | None = None,
    lazy: bool = False,
    **kwargs
) -> Trace
```

**Parameters:**

| Parameter | Type               | Description                                                  |
| --------- | ------------------ | ------------------------------------------------------------ |
| `path`    | `str \|PathLike`   | Path to the file to load                                     |
| `format`  | `str \|None`       | Optional format override (e.g., "tektronix", "rigol", "csv") |
| `channel` | `str \|int \|None` | Optional channel name or index for multi-channel files       |
| `lazy`    | `bool`             | If True, use lazy loading for huge files                     |

**Returns:** `WaveformTrace` or `DigitalTrace` depending on file content.

**Raises:**

- `UnsupportedFormatError`: If the file format is not recognized
- `LoaderError`: If the file cannot be loaded
- `FileNotFoundError`: If the file does not exist

**Example:**

```python
import oscura as osc

# Basic loading
trace = osc.load("oscilloscope_capture.wfm")
print(f"Loaded {len(trace.data)} samples at {trace.metadata.sample_rate} Hz")

# Force specific loader
trace = osc.load("data.bin", format="tektronix")

# Check if digital trace
from oscura.core.types import DigitalTrace
if isinstance(trace, DigitalTrace):
    print("Loaded digital waveform")
```

### `load_all_channels()`

Load all channels from a multi-channel waveform file.

```python
def load_all_channels(
    path: str | PathLike[str],
    *,
    format: str | None = None,
) -> dict[str, WaveformTrace | DigitalTrace]
```

**Parameters:**

| Parameter | Type             | Description                             |
| --------- | ---------------- | --------------------------------------- |
| `path`    | `str \|PathLike` | Path to the multi-channel waveform file |
| `format`  | `str \|None`     | Optional format override                |

**Returns:** Dictionary mapping channel names to traces. Analog channels are named "ch1", "ch2", etc. Digital channels are named "d1", "d2", etc.

**Example:**

```python
import oscura as osc

channels = osc.load_all_channels("multi_channel.wfm")
for name, trace in channels.items():
    print(f"{name}: {len(trace.data)} samples")

# Access specific channel
analog_ch1 = channels["ch1"]
digital_d1 = channels["d1"]
```

### `load_lazy()`

Load trace with lazy loading for huge files.

```python
def load_lazy(
    path: str | PathLike[str],
    **kwargs
) -> LazyWaveformTrace | WaveformTrace
```

**Parameters:**

| Parameter  | Type             | Description                              |
| ---------- | ---------------- | ---------------------------------------- |
| `path`     | `str \|PathLike` | Path to the file                         |
| `**kwargs` | -                | Additional arguments (sample_rate, etc.) |

**Returns:** `LazyWaveformTrace` or `WaveformTrace`.

**Example:**

```python
import oscura as osc

trace = osc.load_lazy("huge_trace.npy", sample_rate=1e9)
print(f"Length: {trace.length}")  # Metadata available immediately
```

### `get_supported_formats()`

Get list of supported file formats.

```python
def get_supported_formats() -> list[str]
```

**Returns:** List of supported file extensions.

**Example:**

```python
from oscura.loaders import get_supported_formats
print(get_supported_formats())
# ['.wfm', '.npz', '.csv', '.h5', '.hdf5', '.sr', '.pcap', '.wav', '.vcd', '.tdms', ...]
```

## Supported Formats

| Extension          | Format                  | Loader         |
| ------------------ | ----------------------- | -------------- |
| `.wfm`             | Tektronix/Rigol WFM     | Auto-detect    |
| `.npz`             | NumPy compressed        | `numpy_loader` |
| `.csv`             | Comma-separated values  | `csv_loader`   |
| `.h5`, `.hdf5`     | HDF5                    | `hdf5_loader`  |
| `.sr`              | Sigrok                  | `sigrok`       |
| `.pcap`, `.pcapng` | Packet capture          | `pcap`         |
| `.wav`             | Audio waveform          | `wav`          |
| `.vcd`             | Value Change Dump       | `vcd`          |
| `.tdms`            | NI TDMS                 | `tdms`         |
| `.s1p` - `.s8p`    | Touchstone S-parameters | `touchstone`   |

## Configurable Binary Loading

For custom binary packet formats, Oscura provides a configurable loader system.

### `ConfigurablePacketLoader`

Load packets from binary files using YAML-defined packet formats.

```python
from oscura.loaders import ConfigurablePacketLoader, PacketFormatConfig

# Define packet format
config = PacketFormatConfig(
    header_fields=[
        HeaderFieldDef(name="sync", dtype="uint16", offset=0),
        HeaderFieldDef(name="length", dtype="uint16", offset=2),
    ],
    sample_format=SampleFormatDef(dtype="int16", samples_per_packet=128),
    packet_size=260,
)

# Load packets
loader = ConfigurablePacketLoader(config)
packets = loader.load("data.bin")
```

### `load_binary_packets()`

Convenience function for loading binary packets.

```python
from oscura.loaders import load_binary_packets

packets = load_binary_packets(
    "data.bin",
    config_path="packet_config.yaml"
)
```

## Preprocessing Functions

### `detect_idle_regions()`

Detect idle regions in waveform data.

```python
from oscura.loaders import detect_idle_regions

regions = detect_idle_regions(trace.data, threshold=0.01)
for region in regions:
    print(f"Idle from {region.start} to {region.end}")
```

### `trim_idle()`

Remove idle regions from waveform data.

```python
from oscura.loaders import trim_idle

trimmed_data = trim_idle(trace.data, threshold=0.01)
```

### `get_idle_statistics()`

Get statistics about idle regions.

```python
from oscura.loaders import get_idle_statistics

stats = get_idle_statistics(trace.data)
print(f"Total idle: {stats.total_idle_samples} samples")
print(f"Idle percentage: {stats.idle_percentage:.1f}%")
```

## Validation

### `PacketValidator`

Validate packet integrity.

```python
from oscura.loaders import PacketValidator

validator = PacketValidator()
result = validator.validate(packets)
print(f"Valid: {result.valid}")
print(f"Errors: {result.errors}")
```

## Data Types

### `WaveformTrace`

Analog waveform trace with metadata.

```python
from oscura.core.types import WaveformTrace

# Properties
trace.data          # numpy array of samples
trace.metadata      # TraceMetadata object
trace.metadata.sample_rate     # Sample rate in Hz
trace.metadata.channel_name    # Channel name
trace.metadata.vertical_scale  # Vertical scale (V/div)
trace.metadata.vertical_offset # Vertical offset (V)
```

### `DigitalTrace`

Digital waveform trace.

```python
from oscura.core.types import DigitalTrace

# Properties
trace.data          # numpy array of digital values
trace.metadata      # TraceMetadata object
trace.bit_width     # Number of bits per sample
```

### `LazyWaveformTrace`

Lazy-loaded waveform for large files.

```python
from oscura.loaders import LazyWaveformTrace

# Properties
trace.length        # Total samples (available without loading)
trace.data          # Loads data on access
trace[0:1000]       # Slice access loads only requested range
```

## Error Handling

```python
from oscura.core.exceptions import LoaderError, UnsupportedFormatError

try:
    trace = osc.load("file.xyz")
except UnsupportedFormatError as e:
    print(f"Unsupported format: {e.extension}")
    print(f"Supported formats: {e.supported_formats}")
except LoaderError as e:
    print(f"Load error: {e.message}")
    print(f"File: {e.file_path}")
    print(f"Fix hint: {e.fix_hint}")
```

## See Also

- [Export API](export.md) - Data export functionality
- [Analysis API](analysis.md) - Analysis functions
