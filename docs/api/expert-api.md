# Expert/Extensibility API Reference

> **Version**: 0.5.0 | **Last Updated**: 2026-01-19

Comprehensive API documentation for extending Oscura with custom algorithms, measurements, and plugins.

## Overview

The Oscura Expert API provides powerful extensibility mechanisms for advanced users who need to:

- Register custom analysis algorithms
- Define domain-specific measurements
- Build reusable plugins
- Extend protocol decoders
- Create custom file format loaders
- Add new export formats

This API is designed for:

- Library developers building on Oscura
- Advanced users with specialized analysis needs
- Plugin authors creating reusable extensions
- Organizations standardizing custom measurements

## Table of Contents

1. [Algorithm Registry](#algorithm-registry)
2. [Custom Measurements](#custom-measurements)
3. [Plugin System](#plugin-system)
4. [Plugin Templates](#plugin-templates)
5. [Extension Points](#extension-points)
6. [Complete Examples](#complete-examples)

---

## Algorithm Registry

### Overview

The `AlgorithmRegistry` allows you to register custom algorithms that integrate seamlessly with Oscura's analysis functions. Algorithms are organized by category (e.g., `edge_detector`, `peak_finder`, `window_func`) and can be called by name throughout Oscura.

### Classes

#### `AlgorithmRegistry`

Singleton registry for custom algorithm implementations.

**Methods:**

| Method                                          | Description                             |
| ----------------------------------------------- | --------------------------------------- |
| `register(name, func, category, validate=True)` | Register a custom algorithm             |
| `get(category, name)`                           | Retrieve algorithm by category and name |
| `list_categories()`                             | List all registered categories          |
| `list_algorithms(category)`                     | List algorithms in a category           |
| `has_algorithm(category, name)`                 | Check if algorithm exists               |
| `unregister(category, name)`                    | Remove algorithm from registry          |
| `clear_category(category)`                      | Clear all algorithms in category        |
| `clear_all()`                                   | Clear all registered algorithms         |

### Functions

#### `register_algorithm()`

Register a custom algorithm in the global registry.

```python
def register_algorithm(
    name: str,
    func: Callable,
    category: str,
    validate: bool = True
) -> None
```

**Parameters:**

- `name` (str): Unique name for the algorithm within its category
- `func` (Callable): Function implementing the algorithm
- `category` (str): Algorithm category (e.g., `'edge_detector'`, `'peak_finder'`)
- `validate` (bool, optional): Whether to validate function signature. Default `True`

**Raises:**

- `ValueError`: If name already exists in category
- `TypeError`: If func is not callable or signature is invalid

**Example:**

```python
import oscura as osc
import numpy as np

# Register custom edge detector
def schmitt_trigger(data, low_threshold=0.3, high_threshold=0.7, **kwargs):
    """Custom Schmitt trigger edge detector."""
    edges = []
    state = data[0] > high_threshold

    for i, val in enumerate(data):
        if not state and val > high_threshold:
            edges.append(i)
            state = True
        elif state and val < low_threshold:
            edges.append(i)
            state = False

    return edges

# Register algorithm
osc.register_algorithm(
    'schmitt_trigger',
    schmitt_trigger,
    category='edge_detector'
)

# Use in edge detection
trace = osc.load("capture.wfm")
edges = osc.find_edges(trace, method='schmitt_trigger', low_threshold=0.2, high_threshold=0.8)
```

#### `get_algorithm()`

Retrieve algorithm from global registry.

```python
def get_algorithm(category: str, name: str) -> Callable
```

**Parameters:**

- `category` (str): Algorithm category
- `name` (str): Algorithm name

**Returns:**

- `Callable`: The registered algorithm function

**Raises:**

- `KeyError`: If category or name not found

**Example:**

```python
import oscura as osc

# Get registered algorithm
edge_detector = osc.get_algorithm('edge_detector', 'schmitt_trigger')

# Call directly
edges = edge_detector(data, low_threshold=0.2, high_threshold=0.8)
```

#### `get_algorithms()`

List all algorithms in a category.

```python
def get_algorithms(category: str) -> list[str]
```

**Parameters:**

- `category` (str): Algorithm category

**Returns:**

- `list[str]`: List of algorithm names in the category

**Example:**

```python
import oscura as osc

# List available edge detectors
detectors = osc.get_algorithms('edge_detector')
print(f"Available edge detectors: {detectors}")
# Output: ['threshold', 'hysteresis', 'schmitt_trigger']
```

### Algorithm Categories

Common algorithm categories:

| Category        | Description                 | Example Algorithms                 |
| --------------- | --------------------------- | ---------------------------------- |
| `edge_detector` | Edge detection algorithms   | `threshold`, `hysteresis`, `canny` |
| `peak_finder`   | Peak detection algorithms   | `simple`, `prominence`, `cwt`      |
| `window_func`   | Window functions for FFT    | `hanning`, `hamming`, `blackman`   |
| `filter`        | Signal filtering algorithms | `lowpass`, `highpass`, `bandpass`  |
| `interpolator`  | Interpolation methods       | `linear`, `cubic`, `spline`        |
| `preprocessor`  | Signal preprocessing        | `detrend`, `normalize`, `clip`     |

---

## Custom Measurements

### Overview

The custom measurement framework allows you to define domain-specific measurements that integrate with Oscura's batch processing and export capabilities. Measurements include metadata (units, category, description) and can be filtered by tags.

### Classes

#### `MeasurementDefinition`

Definition of a custom measurement with metadata.

**Attributes:**

- `name` (str): Unique name for the measurement
- `func` (Callable): Function that computes the measurement
- `units` (str): Units of measurement (e.g., `'V'`, `'Hz'`, `'s'`, `'ratio'`)
- `category` (str): Measurement category (e.g., `'amplitude'`, `'timing'`, `'frequency'`)
- `description` (str): Human-readable description
- `tags` (list[str]): Optional tags for categorization and search

**Example:**

```python
import oscura as osc
import numpy as np

def calculate_crest_factor(trace, **kwargs):
    """Calculate crest factor: peak / RMS."""
    peak = abs(trace.data).max()
    rms = np.sqrt((trace.data ** 2).mean())
    return peak / rms if rms > 0 else 0.0

# Create measurement definition
crest_factor_defn = osc.MeasurementDefinition(
    name='crest_factor',
    func=calculate_crest_factor,
    units='ratio',
    category='amplitude',
    description='Crest factor (peak/RMS ratio)',
    tags=['amplitude', 'quality', 'waveform']
)
```

#### `MeasurementRegistry`

Registry for custom measurements.

**Methods:**

| Method                                        | Description                        |
| --------------------------------------------- | ---------------------------------- |
| `register(name, func, units, category, ...)`  | Register a custom measurement      |
| `get(name)`                                   | Get measurement definition by name |
| `has_measurement(name)`                       | Check if measurement exists        |
| `list_measurements(category=None, tags=None)` | List registered measurements       |
| `get_metadata(name)`                          | Get metadata for a measurement     |
| `unregister(name)`                            | Remove measurement from registry   |

### Functions

#### `register_measurement()`

Register a custom measurement in the global registry.

```python
def register_measurement(
    name: str = None,
    func: Callable = None,
    units: str = None,
    category: str = None,
    description: str = "",
    tags: list[str] = None,
    definition: MeasurementDefinition = None
) -> None
```

**Parameters:**

- `name` (str): Measurement name (required if definition not provided)
- `func` (Callable): Measurement function accepting `(trace, **kwargs) -> float`
- `units` (str): Units of measurement
- `category` (str): Measurement category
- `description` (str, optional): Description of measurement
- `tags` (list[str], optional): Tags for categorization
- `definition` (MeasurementDefinition, optional): Pre-built definition

**Raises:**

- `ValueError`: If required parameters missing or name already exists

**Example:**

```python
import oscura as osc
import numpy as np

# Define measurement function
def slew_rate_max(trace, **kwargs):
    """Calculate maximum slew rate."""
    dt = 1.0 / trace.metadata.sample_rate
    derivative = np.diff(trace.data) / dt
    return abs(derivative).max()

# Register measurement
osc.register_measurement(
    name='max_slew_rate',
    func=slew_rate_max,
    units='V/s',
    category='edge',
    description='Maximum slew rate in trace',
    tags=['edge', 'derivative', 'speed']
)

# Use in analysis
trace = osc.load("capture.wfm")
slew = osc.measure_custom(trace, 'max_slew_rate')
print(f"Max slew rate: {slew:.2e} V/s")
```

#### `measure()`

Execute a registered measurement.

```python
def measure(trace: WaveformTrace, name: str, **kwargs) -> float
```

**Parameters:**

- `trace` (WaveformTrace): Trace to measure
- `name` (str): Measurement name
- `**kwargs`: Additional parameters for the measurement

**Returns:**

- `float`: Measured value

**Raises:**

- `KeyError`: If measurement not found

**Example:**

```python
import oscura as osc

trace = osc.load("capture.wfm")

# Execute registered measurement
cf = osc.measure_custom(trace, 'crest_factor')
slew = osc.measure_custom(trace, 'max_slew_rate')

print(f"Crest factor: {cf:.2f}")
print(f"Max slew rate: {slew:.2e} V/s")
```

#### `list_measurements()`

List registered measurements with optional filtering.

```python
def list_measurements(
    category: str = None,
    tags: list[str] = None
) -> list[str]
```

**Parameters:**

- `category` (str, optional): Filter by category
- `tags` (list[str], optional): Filter by tags

**Returns:**

- `list[str]`: List of measurement names

**Example:**

```python
import oscura as osc

# List all measurements
all_measurements = osc.list_measurements()

# List amplitude measurements
amplitude = osc.list_measurements(category='amplitude')

# List measurements with 'edge' tag
edge_measurements = osc.list_measurements(tags=['edge'])

print(f"Amplitude measurements: {amplitude}")
```

#### `get_measurement_registry()`

Get the global measurement registry for advanced operations.

```python
def get_measurement_registry() -> MeasurementRegistry
```

**Returns:**

- `MeasurementRegistry`: Global registry instance

**Example:**

```python
import oscura as osc

# Get registry
registry = osc.get_measurement_registry()

# Get detailed metadata
metadata = registry.get_metadata('crest_factor')
print(f"Name: {metadata['name']}")
print(f"Units: {metadata['units']}")
print(f"Category: {metadata['category']}")
print(f"Description: {metadata['description']}")
print(f"Tags: {metadata['tags']}")
```

### Measurement Categories

Common measurement categories:

| Category      | Description                   | Examples                              |
| ------------- | ----------------------------- | ------------------------------------- |
| `amplitude`   | Amplitude measurements        | `peak`, `rms`, `crest_factor`         |
| `timing`      | Time-domain measurements      | `period`, `frequency`, `duty_cycle`   |
| `edge`        | Edge-related measurements     | `rise_time`, `fall_time`, `slew_rate` |
| `frequency`   | Frequency-domain measurements | `thd`, `snr`, `sinad`                 |
| `quality`     | Signal quality metrics        | `enob`, `sfdr`, `snr_estimate`        |
| `statistical` | Statistical measurements      | `mean`, `std`, `skewness`             |

---

## Plugin System

### Overview

Oscura's plugin system enables discovery and loading of third-party extensions via Python entry points. Plugins can provide protocol decoders, custom measurements, file format loaders, and export handlers.

### Classes

#### `PluginMetadata`

Metadata about a loaded plugin.

**Attributes:**

- `name` (str): Plugin name
- `entry_point` (str): Entry point group
- `version` (str): Plugin version (if available)
- `module` (str): Module name
- `callable` (Any): The loaded plugin object
- `dependencies` (list[str]): Plugin dependencies (if available)

**Example:**

```python
import oscura as osc

# Load plugin
plugin = osc.load_plugin('oscura.decoders', 'custom_decoder')

print(f"Loaded: {plugin.name} v{plugin.version}")
print(f"Module: {plugin.module}")
```

#### `PluginManager`

Manager for discovering and loading third-party plugins.

**Entry Point Groups:**

- `oscura.decoders`: Protocol decoders
- `oscura.measurements`: Custom measurements
- `oscura.loaders`: File format loaders
- `oscura.exporters`: Export format handlers

**Methods:**

| Method                                   | Description                                 |
| ---------------------------------------- | ------------------------------------------- |
| `discover_plugins(group=None)`           | Discover available plugins via entry points |
| `load_plugin(group, name, reload=False)` | Load a plugin by group and name             |
| `get_plugin(group, name)`                | Get loaded plugin callable                  |
| `is_loaded(group, name)`                 | Check if plugin is already loaded           |
| `list_loaded_plugins()`                  | List all loaded plugins                     |
| `unload_plugin(group, name)`             | Unload plugin from cache                    |

**Example:**

```python
import oscura as osc

# Get plugin manager
manager = osc.get_plugin_manager()

# Discover all available plugins
plugins = manager.discover_plugins()
print(f"Available decoders: {plugins['oscura.decoders']}")

# Load specific plugin
plugin = manager.load_plugin('oscura.decoders', 'flexray')

# Use plugin
decoder = plugin.callable
frames = decoder.decode(trace)
```

### Functions

#### `load_plugin()`

Load a plugin from the global plugin manager.

```python
def load_plugin(group: str, name: str) -> PluginMetadata
```

**Parameters:**

- `group` (str): Entry point group
- `name` (str): Plugin name

**Returns:**

- `PluginMetadata`: Loaded plugin metadata

**Raises:**

- `PluginError`: If plugin fails to load

**Example:**

```python
import oscura as osc

try:
    plugin = osc.load_plugin('oscura.decoders', 'flexray')
    print(f"Loaded {plugin.name} v{plugin.version}")

    # Use plugin
    decoder = plugin.callable(baudrate=10000000)
    frames = decoder.decode(trace)

except osc.PluginError as e:
    print(f"Plugin failed to load: {e}")
```

#### `list_plugins()`

List available plugins.

```python
def list_plugins(group: str = None) -> dict[str, list[str]]
```

**Parameters:**

- `group` (str, optional): Specific group to list. If None, lists all groups

**Returns:**

- `dict[str, list[str]]`: Dictionary mapping group names to plugin names

**Example:**

```python
import oscura as osc

# List all available plugins
plugins = osc.list_plugins()

for group, plugin_list in plugins.items():
    print(f"{group}:")
    for plugin in plugin_list:
        print(f"  - {plugin}")

# List only decoders
decoders = osc.list_plugins('oscura.decoders')
print(f"Available decoders: {decoders['oscura.decoders']}")
```

#### `get_plugin_manager()`

Get the global plugin manager instance.

```python
def get_plugin_manager() -> PluginManager
```

**Returns:**

- `PluginManager`: Global plugin manager

**Example:**

```python
import oscura as osc

manager = osc.get_plugin_manager()

# List loaded plugins
loaded = manager.list_loaded_plugins()
for plugin in loaded:
    print(f"{plugin.name} v{plugin.version} ({plugin.entry_point})")
```

### Creating Plugin Packages

To make your extension discoverable as a plugin, add entry points to your `pyproject.toml`:

```toml
[project.entry-points."oscura.decoders"]
flexray = "my_package.flexray:FlexRayDecoder"

[project.entry-points."oscura.measurements"]
custom_snr = "my_package.measurements:snr_measurement"

[project.entry-points."oscura.loaders"]
custom_format = "my_package.loaders:CustomLoader"
```

After installation (`pip install my-package`), Oscura will automatically discover these plugins.

---

## Plugin Templates

### Overview

Oscura provides template generation tools to scaffold new plugins with all necessary boilerplate, tests, and documentation.

### Types

#### `PluginType`

Plugin type definitions.

```python
PluginType = Literal["analyzer", "loader", "exporter", "decoder"]
```

### Classes

#### `PluginTemplate`

Configuration for plugin template generation.

**Attributes:**

- `name` (str): Plugin name (e.g., `'my_custom_decoder'`)
- `plugin_type` (PluginType): Type of plugin
- `output_dir` (Path): Directory where plugin will be generated
- `author` (str): Plugin author name
- `description` (str): Brief description of plugin functionality
- `version` (str): Initial plugin version (default: `'0.1.0'`)

**Example:**

```python
from pathlib import Path
import oscura as osc

template = osc.PluginTemplate(
    name='flexray_decoder',
    plugin_type='decoder',
    output_dir=Path('plugins/flexray'),
    author='John Doe',
    description='FlexRay protocol decoder'
)
```

### Functions

#### `generate_plugin_template()`

Generate a plugin skeleton with all necessary boilerplate.

```python
def generate_plugin_template(
    name: str,
    plugin_type: PluginType,
    output_dir: Path,
    *,
    author: str = "Plugin Author",
    description: str = None,
    version: str = "0.1.0"
) -> Path
```

**Parameters:**

- `name` (str): Plugin name (converted to snake_case)
- `plugin_type` (PluginType): Type of plugin to generate
- `output_dir` (Path): Directory where plugin will be created
- `author` (str, optional): Plugin author name
- `description` (str, optional): Plugin description (auto-generated if None)
- `version` (str, optional): Initial plugin version

**Returns:**

- `Path`: Path to the generated plugin directory

**Raises:**

- `ValueError`: If plugin_type is invalid
- `FileExistsError`: If output_dir already exists

**Generated Structure:**

```
plugins/flexray/
├── __init__.py           # Plugin metadata and entry point
├── flexray_decoder.py    # Main implementation
├── tests/
│   ├── __init__.py
│   └── test_flexray_decoder.py
├── README.md             # Usage documentation
└── pyproject.toml        # Packaging configuration
```

**Example:**

```python
from pathlib import Path
import oscura as osc

# Generate decoder plugin
plugin_dir = osc.generate_plugin_template(
    name='flexray_decoder',
    plugin_type='decoder',
    output_dir=Path('plugins/flexray'),
    author='John Doe',
    description='FlexRay protocol decoder with full CRC support'
)

print(f"Plugin generated at: {plugin_dir}")

# Install plugin in development mode
# cd plugins/flexray
# pip install -e .
```

### Plugin Types

| Type       | Description        | Use Cases                                            |
| ---------- | ------------------ | ---------------------------------------------------- |
| `decoder`  | Protocol decoder   | UART, SPI, I2C, CAN, FlexRay, custom protocols       |
| `analyzer` | Signal analyzer    | FFT analysis, pattern detection, custom metrics      |
| `loader`   | File format loader | Proprietary oscilloscope formats, custom data files  |
| `exporter` | Export handler     | Custom export formats, database export, cloud upload |

---

## Extension Points

### Overview

Extension points provide hooks for modifying Oscura behavior at specific points in the processing pipeline. This is an advanced feature for deep integration.

### Classes

#### `ExtensionPointSpec`

Specification for an extension point.

**Attributes:**

- `name` (str): Extension point name
- `description` (str): Description of what the extension point does
- `hook_signature` (str): Expected function signature
- `priority_enabled` (bool): Whether hooks can have priority
- `allow_multiple` (bool): Whether multiple hooks can be registered

#### `HookContext`

Context passed to extension point hooks.

**Attributes:**

- `extension_point` (str): Name of extension point
- `args` (tuple): Positional arguments
- `kwargs` (dict): Keyword arguments
- `result` (Any): Result from previous hook (if chaining)

### Functions

#### `register_extension_point()`

Register a new extension point.

```python
def register_extension_point(
    name: str,
    description: str,
    hook_signature: str,
    priority_enabled: bool = True,
    allow_multiple: bool = True
) -> None
```

**Example:**

```python
import oscura as osc

# Register extension point for pre-processing
osc.register_extension_point(
    name='preprocess_trace',
    description='Hook for preprocessing traces before analysis',
    hook_signature='(trace: WaveformTrace) -> WaveformTrace',
    priority_enabled=True,
    allow_multiple=True
)
```

#### `hook()`

Decorator to register a function at an extension point.

```python
def hook(
    extension_point: str,
    priority: int = 50,
    error_policy: str = 'raise'
)
```

**Parameters:**

- `extension_point` (str): Extension point name
- `priority` (int, optional): Hook priority (lower = earlier). Default 50
- `error_policy` (str, optional): How to handle errors: `'raise'`, `'log'`, `'ignore'`

**Example:**

```python
import oscura as osc

@osc.hook('preprocess_trace', priority=10)
def remove_dc_offset(trace):
    """Remove DC offset before analysis."""
    mean = trace.data.mean()
    trace.data = trace.data - mean
    return trace

@osc.hook('preprocess_trace', priority=20)
def normalize_amplitude(trace):
    """Normalize amplitude to [-1, 1]."""
    max_val = abs(trace.data).max()
    if max_val > 0:
        trace.data = trace.data / max_val
    return trace
```

---

## Complete Examples

### Example 1: Custom Measurement Algorithm

Create a custom measurement for signal-to-noise ratio estimation.

```python
import numpy as np
import oscura as osc

def snr_estimate(trace, **kwargs):
    """
    Estimate SNR using high-frequency noise estimation.

    Method:
    1. Estimate noise from high-frequency components
    2. Calculate signal RMS
    3. Return SNR in dB
    """
    # Estimate noise from differences (high-freq content)
    diff = np.diff(trace.data)
    noise_rms = np.sqrt(np.mean(diff**2)) / np.sqrt(2)

    # Calculate signal RMS
    signal_rms = np.sqrt(np.mean(trace.data**2))

    # Return SNR in dB
    if noise_rms == 0:
        return float('inf')

    return 20 * np.log10(signal_rms / noise_rms)

# Register measurement
osc.register_measurement(
    name='snr_estimate',
    func=snr_estimate,
    units='dB',
    category='quality',
    description='SNR estimated from high-frequency noise',
    tags=['quality', 'noise', 'snr']
)

# Use in analysis
trace = osc.load("noisy_signal.wfm")
snr = osc.measure_custom(trace, 'snr_estimate')
print(f"Estimated SNR: {snr:.1f} dB")
```

### Example 2: Custom Edge Detector

Register a custom edge detection algorithm with hysteresis.

```python
import numpy as np
import oscura as osc

def hysteresis_edge_detector(data, low_threshold=0.3, high_threshold=0.7, **kwargs):
    """
    Edge detector with hysteresis for noise immunity.

    Parameters:
        data: Signal data
        low_threshold: Lower threshold for edge release
        high_threshold: Upper threshold for edge trigger

    Returns:
        List of edge indices
    """
    edges = []
    state = 'low'  # Start in low state

    for i, val in enumerate(data):
        if state == 'low' and val > high_threshold:
            edges.append(i)
            state = 'high'
        elif state == 'high' and val < low_threshold:
            edges.append(i)
            state = 'low'

    return edges

# Register algorithm
osc.register_algorithm(
    name='hysteresis',
    func=hysteresis_edge_detector,
    category='edge_detector'
)

# Use in edge detection
trace = osc.load("digital_signal.wfm")
edges = osc.find_edges(
    trace,
    method='hysteresis',
    low_threshold=0.2,
    high_threshold=0.8
)

print(f"Found {len(edges)} edges")
print(f"Edge times: {edges[:5]} ...")  # First 5 edges
```

### Example 3: Building a Oscura Plugin

Create a complete plugin for a custom protocol decoder.

**Step 1: Generate plugin template**

```python
from pathlib import Path
import oscura as osc

# Generate plugin skeleton
plugin_dir = osc.generate_plugin_template(
    name='modbus_decoder',
    plugin_type='decoder',
    output_dir=Path('plugins/modbus'),
    author='Your Name',
    description='Modbus RTU protocol decoder'
)
```

**Step 2: Implement decoder (edit `modbus_decoder.py`)**

```python
"""Modbus RTU protocol decoder.

This decoder implements Modbus RTU protocol decoding for Oscura.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ModbusDecoder:
    """Modbus RTU protocol decoder.

    Attributes:
        sample_rate: Sample rate of input signal in Hz
        baudrate: Modbus baudrate (default 9600)
    """

    def __init__(self, *, sample_rate: float = 1_000_000.0, baudrate: int = 9600) -> None:
        """Initialize decoder.

        Args:
            sample_rate: Sample rate in Hz
            baudrate: Modbus baudrate
        """
        self.sample_rate = sample_rate
        self.baudrate = baudrate
        self.samples_per_bit = int(sample_rate / baudrate)

    def decode(self, signal: NDArray[np.uint8]) -> list[dict[str, object]]:
        """Decode Modbus RTU frames from digital signal.

        Args:
            signal: Digital signal (0/1 values)

        Returns:
            List of decoded frames with:
                - address: Device address (1 byte)
                - function: Function code (1 byte)
                - data: Data bytes
                - crc: CRC-16 (2 bytes)
                - valid: CRC validation result
        """
        if len(signal) == 0:
            raise ValueError("Signal cannot be empty")

        frames = []
        # Implement Modbus RTU decoding logic here
        # - Find frame start (3.5 character silence)
        # - Decode bytes (8N1 format)
        # - Extract address, function, data, CRC
        # - Validate CRC-16

        return frames

    def configure(self, **params: object) -> None:
        """Configure decoder parameters.

        Args:
            **params: Decoder-specific parameters
        """
        for key, value in params.items():
            setattr(self, key, value)
```

**Step 3: Install plugin**

```bash
cd plugins/modbus
pip install -e .
```

**Step 4: Use plugin**

```python
import oscura as osc

# Plugin is auto-discovered
trace = osc.load("modbus_capture.wfm")

# Load and use plugin
plugin = osc.load_plugin('oscura.decoders', 'modbus_decoder')
decoder = plugin.callable(sample_rate=1e6, baudrate=9600)

frames = decoder.decode(trace.data)

for frame in frames:
    print(f"Address: 0x{frame['address']:02X}")
    print(f"Function: 0x{frame['function']:02X}")
    print(f"Data: {frame['data']}")
    print(f"CRC Valid: {frame['valid']}")
```

### Example 4: Custom File Format Loader

Create a plugin for loading a proprietary oscilloscope format.

```python
from pathlib import Path
import numpy as np
import oscura as osc

# Generate loader plugin
plugin_dir = osc.generate_plugin_template(
    name='custom_scope_loader',
    plugin_type='loader',
    output_dir=Path('plugins/custom_scope'),
    author='Your Name',
    description='Loader for CustomScope proprietary format'
)

# Implement loader (in custom_scope_loader.py)
class CustomScopeLoader:
    """Loader for CustomScope .cso files."""

    def load(self, file_path: Path) -> dict[str, np.ndarray]:
        """Load data from CustomScope file.

        Args:
            file_path: Path to .cso file

        Returns:
            Dictionary mapping channel names to signal arrays
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read proprietary format
        with file_path.open('rb') as f:
            # Parse header
            magic = f.read(4)
            if magic != b'CSO1':
                raise ValueError("Invalid CustomScope file")

            num_channels = int.from_bytes(f.read(2), 'little')
            sample_rate = float(np.frombuffer(f.read(8), dtype=np.float64)[0])

            # Read channels
            data = {}
            for i in range(num_channels):
                name_len = int.from_bytes(f.read(1), 'little')
                name = f.read(name_len).decode('utf-8')

                num_samples = int.from_bytes(f.read(4), 'little')
                samples = np.frombuffer(f.read(num_samples * 4), dtype=np.float32)

                data[name] = samples

        return data

    @staticmethod
    def can_load(file_path: Path) -> bool:
        """Check if this loader can handle the file."""
        if file_path.suffix != '.cso':
            return False

        # Check magic number
        try:
            with file_path.open('rb') as f:
                magic = f.read(4)
                return magic == b'CSO1'
        except Exception:
            return False

# Install and use
# pip install -e plugins/custom_scope

# Use in Oscura
loader_plugin = osc.load_plugin('oscura.loaders', 'custom_scope_loader')
loader = loader_plugin.callable()

channels = loader.load(Path('capture.cso'))
trace = osc.WaveformTrace(
    data=channels['CH1'],
    metadata=osc.TraceMetadata(sample_rate=1e6)
)
```

### Example 5: Extending Protocol Decoders

Add custom decoding for application-layer protocol on top of UART.

```python
import oscura as osc
from oscura.analyzers.protocols import UARTDecoder

class CustomProtocolDecoder:
    """Decoder for custom protocol over UART."""

    def __init__(self, baudrate: int = 115200):
        self.uart = UARTDecoder(baudrate=baudrate)
        self.baudrate = baudrate

    def decode(self, trace):
        """Decode custom protocol messages.

        Returns:
            List of decoded messages with:
                - type: Message type
                - payload: Decoded payload
                - timestamp: Message timestamp
        """
        # First decode UART bytes
        uart_messages = self.uart.decode(trace)

        # Parse custom protocol
        messages = []
        for msg in uart_messages:
            # Custom protocol format:
            # [START] [TYPE] [LEN] [PAYLOAD...] [CRC]
            data = msg['data']

            if data[0] != 0xAA:  # START byte
                continue

            msg_type = data[1]
            msg_len = data[2]
            payload = data[3:3+msg_len]
            crc = data[3+msg_len]

            # Validate CRC
            calculated_crc = sum(data[1:3+msg_len]) & 0xFF
            if calculated_crc != crc:
                continue

            messages.append({
                'type': msg_type,
                'payload': payload,
                'timestamp': msg['timestamp'],
                'valid': True
            })

        return messages

# Register as algorithm
def custom_protocol_decode(trace, baudrate=115200, **kwargs):
    """Decode custom protocol from trace."""
    decoder = CustomProtocolDecoder(baudrate=baudrate)
    return decoder.decode(trace)

osc.register_algorithm(
    'custom_protocol',
    custom_protocol_decode,
    category='protocol_decoder'
)

# Use in analysis
trace = osc.load("protocol_capture.wfm")
decoder = CustomProtocolDecoder(baudrate=115200)
messages = decoder.decode(trace)

for msg in messages:
    print(f"Type: 0x{msg['type']:02X}")
    print(f"Payload: {msg['payload'].hex()}")
    print(f"Time: {msg['timestamp']:.6f} s")
```

---

## Best Practices

### Algorithm Registration

1. **Use descriptive names**: Choose clear, unique names that describe what the algorithm does
2. **Document parameters**: Include docstrings with parameter descriptions
3. **Accept kwargs**: Use `**kwargs` for extensibility even if not immediately needed
4. **Handle edge cases**: Check for empty data, invalid inputs, division by zero
5. **Return consistent types**: Ensure return type matches expected output format

### Custom Measurements

1. **Include metadata**: Always provide units, category, description, and tags
2. **Use standard units**: Stick to SI units where applicable (V, A, Hz, s)
3. **Document assumptions**: Note any assumptions about signal characteristics
4. **Handle errors gracefully**: Return NaN or raise clear exceptions for invalid inputs
5. **Test thoroughly**: Write unit tests for edge cases and typical usage

### Plugin Development

1. **Follow naming conventions**: Use snake_case for module names, PascalCase for classes
2. **Provide comprehensive docs**: Include README with installation and usage instructions
3. **Include tests**: Write unit tests in `tests/` directory
4. **Version carefully**: Use semantic versioning (MAJOR.MINOR.PATCH)
5. **Declare dependencies**: List all required packages in `pyproject.toml`
6. **Handle errors**: Catch and report errors clearly to help users debug
7. **Document entry points**: Clearly document which entry point group to use

### Performance Considerations

1. **Vectorize operations**: Use NumPy vectorized operations instead of loops
2. **Minimize memory allocation**: Reuse arrays when possible
3. **Profile hot paths**: Use profiling tools to identify bottlenecks
4. **Consider lazy evaluation**: For large datasets, consider streaming/chunked processing
5. **Cache expensive computations**: Store intermediate results when appropriate

---

## See Also

- [API Reference](index.md) - Complete API documentation

---

## API Support

For questions, bug reports, or feature requests:

- **GitHub Issues**: [https://github.com/your-org/oscura/issues](https://github.com/your-org/oscura/issues)
- **Documentation**: [https://oscura.readthedocs.io](https://oscura.readthedocs.io)
- **Advanced Examples**: `demos/14_advanced_inference/` and `demos/15_complete_workflows/` in repository
