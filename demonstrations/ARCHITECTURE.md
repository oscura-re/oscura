# Demonstration Framework Architecture

This document explains how the Oscura demonstration system is designed and organized. It provides guidance for contributors building new demonstrations and maintainers enhancing the framework.

## Overview

The demonstration framework serves as:

- **Validation layer**: Comprehensive testing of all Oscura capabilities
- **Documentation medium**: Working code examples showing how to use the API
- **Coverage tracker**: Automated analysis of which APIs are demonstrated
- **Integration testing**: End-to-end workflows demonstrating real-world scenarios

Currently, the framework includes **112 demonstrations** organized into 19 categories, collectively demonstrating 100+ API capabilities.

---

## BaseDemo Template Pattern

### Core Concept

All demonstrations inherit from `BaseDemo`, a template class that enforces consistency and provides standard infrastructure. This pattern ensures:

- Uniform execution framework across all 112 demonstrations
- Automatic timing measurements and error handling
- Standardized formatting and output
- Built-in validation infrastructure
- CLI argument support (`--data-file` for custom data)

### Class Hierarchy

```python
class BaseDemo(ABC):
    """Base class for all Oscura demonstrations."""

    def __init__(self, name, description, capabilities, ieee_standards, related_demos):
        """Initialize with metadata."""

    def execute(self) -> bool:
        """Main entry point - orchestrates entire workflow."""

    @abstractmethod
    def generate_test_data(self) -> dict:
        """Create synthetic test data."""

    @abstractmethod
    def run_demonstration(self, data: dict) -> dict:
        """Execute demonstration logic."""

    @abstractmethod
    def validate(self, results: dict) -> bool:
        """Validate results against expected values."""
```

### Lifecycle

Each demonstration follows a strict lifecycle:

1. **Initialize** (`__init__`) - Define metadata
   - Demo name and description
   - Capabilities being demonstrated
   - Applicable IEEE standards
   - Related demonstrations

2. **Execute** (`execute()`) - Main orchestrator
   - Parses command-line arguments
   - Manages timing measurements
   - Handles exceptions and keyboard interrupts
   - Prints formatted output

3. **Generate Test Data** (`generate_test_data()`) - Create synthetic data
   - Self-contained generation (no external files)
   - Deterministic (seeded random numbers)
   - Realistic parameters matching real-world scenarios
   - Returns dict of datasets for demonstration

4. **Run Demonstration** (`run_demonstration()`) - Execute logic
   - Uses BaseDemo formatting methods (section, info, result, etc.)
   - Calls Oscura APIs to perform measurements
   - Returns dict of results for validation

5. **Validate** (`validate()`) - Verify correctness
   - Uses validation helpers to verify results
   - Returns True/False pass/fail status
   - Uses tolerance levels from constants.py

### Why This Pattern?

**Consistency**: Same structure across all 112 demonstrations makes them easier to learn from and maintain.

**Reusability**: Common utilities (data generation, validation, formatting) prevent code duplication.

**Validation Framework**: Automatic infrastructure for testing every demonstrated feature.

**Extensibility**: New demonstrations only need to implement 3 abstract methods.

---

## Validation Framework

### Core Validation Helpers

Located in `demonstrations/common/validation.py`:

#### `validate_approximately(actual, expected, tolerance, name)`

Validates floating-point values within a relative tolerance.

```python
from demonstrations.common import validate_approximately

# Validate frequency measurement
if not validate_approximately(
    actual_freq,
    expected_freq=1000.0,
    tolerance=0.01,  # ±1%
    name="Frequency"
):
    return False
```

**When to use**: Analog measurements (frequency, amplitude, timing) where inherent measurement uncertainty exists.

#### `validate_range(value, min_val, max_val, name)`

Validates that a value falls within a specified range.

```python
from demonstrations.common import validate_range

# Validate voltage is between 4.5V and 5.5V
if not validate_range(vdd, min_val=4.5, max_val=5.5, name="Supply Voltage"):
    return False
```

**When to use**: Bounded measurements where both upper and lower limits matter.

#### `validate_results(results, expected)`

Batch validation comparing multiple results against expected values.

```python
from demonstrations.common import validate_results

expected = {
    "amplitude": {"min": 1.95, "max": 2.05},
    "frequency": 1000.0,
    "duty_cycle": {"min": 0.48, "max": 0.52},
}
return validate_results(results, expected)
```

**When to use**: Complex demonstrations with multiple measurements requiring coordinated validation.

#### Additional Helpers

- `validate_range(value, min_val, max_val, name)` - Bounded range validation
- `validate_exists(obj, name)` - Non-None check
- `validate_length(seq, expected_length, name)` - Sequence length validation
- `validate_type(obj, expected_type, name)` - Type checking

### Tolerance Levels (SSOT)

Tolerances are defined in `demonstrations/common/constants.py`:

```python
TOLERANCE_STRICT = 0.01      # 1% - Precise measurements (clocks, references)
TOLERANCE_NORMAL = 0.05      # 5% - Typical measurements (amplitude, frequency)
TOLERANCE_RELAXED = 0.10     # 10% - Noisy measurements (jitter, harmonics)
```

**Best Practice**: Choose tolerance based on measurement type:

- **Strict (1%)** - Digital clock frequencies, reference signals, mathematically exact calculations
- **Normal (5%)** - Typical analog measurements (voltage, frequency derived from signals)
- **Relaxed (10%)** - Noisy measurements, harmonic content, jitter, estimated parameters

### Validation Best Practices

1. **Derive expected values from generation parameters**
   ```python
   # Generate signal with known parameters
   freq_hz = 1000.0
   amp_v = 1.0
   signal = generate_sine_wave(frequency=freq_hz, amplitude=amp_v)

   # Validation uses these same parameters
   return validate_approximately(measured_freq, freq_hz, tolerance=TOLERANCE_NORMAL)
   ```

2. **Account for measurement uncertainty**
   ```python
   # Digital measurement on analog signal has inherent quantization error
   # Use relaxed tolerance for derived measurements
   if not validate_approximately(thd, expected_thd, tolerance=TOLERANCE_RELAXED):
       return False
   ```

3. **Validate multiple aspects of results**
   ```python
   # Check both presence and correctness
   if "frequency" not in results:
       self.error("Missing frequency result")
       return False

   if not validate_approximately(results["frequency"], expected, tolerance=0.05):
       return False
   ```

---

## Data Generation Principles

### Self-Contained Generation

All demonstrations generate synthetic test data using `generate_test_data()`:

```python
def generate_test_data(self) -> dict:
    """Generate test data without external files."""

    # ✓ GOOD: Uses data generation utilities
    trace = generate_sine_wave(
        frequency=1000.0,
        amplitude=1.0,
        duration=0.001,
        sample_rate=100e3,
    )
    return {"trace": trace}

    # ✗ BAD: Loads from external file
    # with open("signal_data.npz") as f:
    #     return np.load(f)
```

**Why**: Makes demonstrations:
- Reproducible without managing external files
- Fast (no I/O overhead)
- Self-documenting (parameters show intent)
- Maintainable (no file format dependencies)

### Deterministic Random Numbers

Always seed random number generators for reproducibility:

```python
import numpy as np
from demonstrations.common import constants

# Seed for deterministic generation
np.random.seed(constants.RANDOM_SEED)

# Generate noise with known statistical properties
noise = np.random.normal(0, sigma, num_samples)
```

**SSOT**: Use `constants.RANDOM_SEED` (currently 42) for all random initialization.

### Realistic Parameters

Generation parameters should match real-world scenarios:

```python
# ✓ GOOD: Realistic automotive CAN bus parameters
baudrate = 500_000  # 500 kbps standard
sample_rate = 10e6  # 10 MHz typical
voltage = 5.0       # 5V CAN logic

# ✗ BAD: Unrealistic parameters
baudrate = 1_000_000_000  # Not practical
sample_rate = 1e12        # Not achievable
voltage = 12.3            # Arbitrary precision
```

### Documentation in Code

Use docstrings to document generation parameters:

```python
def generate_test_data(self) -> dict:
    """Generate test signals for spectral analysis.

    Generates:
    1. Pure sine: 1kHz fundamental at 1V peak
    2. With harmonics: 1kHz + 3rd harmonic at -20dB
    3. With noise: Added Gaussian noise at 40dB SNR

    Parameters match IEEE 1241-2010 ADC test scenarios.
    """
```

---

## Capability Tracking System

### Purpose

The capability index automatically tracks which Oscura APIs are demonstrated:

- **API coverage matrix**: Which demonstrations use which capabilities
- **Module coverage report**: Which modules are well-covered
- **Gap analysis**: Which capabilities lack demonstrations
- **Cross-reference index**: Find demonstrations by capability

### How Capabilities Are Registered

Demonstrations declare capabilities in `__init__()`:

```python
class MyDemo(BaseDemo):
    def __init__(self):
        super().__init__(
            name="my_demo",
            description="...",
            capabilities=[
                "oscura.fft",           # Import from oscura
                "oscura.psd",           # Method calls to oscura
                "oscura.window.blackman",  # Nested module access
            ],
            ieee_standards=["IEEE 1241-2010"],
            related_demos=[
                "02_basic_analysis/01_waveform_measurements.py",
            ],
        )
```

### Capability Index Usage

Generate capability index:

```bash
# Print to console
python demonstrations/capability_index.py

# Save as markdown
python demonstrations/capability_index.py --output INDEX.md

# Show only coverage gaps
python demonstrations/capability_index.py --gaps-only
```

### Output Format

The index generates reports showing:

- **Summary statistics**: Total demonstrations, capabilities, API coverage %
- **Demonstrations by section**: Organized by category (00_getting_started, etc.)
- **Coverage gaps**: APIs without demonstrations

Example:
```
Total Demonstrations: 112
Capabilities Demonstrated: 48
API Symbols in __all__: 52
API Symbols Demonstrated: 48
API Coverage: 92.3%

Coverage Gaps:
- oscura.experimental.advanced_signal_fusion
- oscura.contrib.legacy_format
```

### Adding New Capability Tracking

1. Add capability name to `capabilities` list in `__init__()`:
   ```python
   capabilities=["oscura.new_feature", "oscura.new_feature.submodule"]
   ```

2. Run capability indexer to verify:
   ```bash
   python demonstrations/capability_index.py --gaps-only
   ```

3. Capabilities are extracted automatically via AST parsing of imports and calls.

---

## Common Utilities (demonstrations/common/)

### BaseDemo (`base_demo.py`)

Core template class providing:

**Initialization**:
- `__init__(name, description, capabilities, ieee_standards, related_demos)` - Metadata setup

**Execution**:
- `execute()` - Main orchestrator (handles lifecycle, timing, error handling)
- `load_custom_data(data_file)` - Load NPZ files for custom data experimentation

**Formatting**:
- `section(title)` - Print major section header (with borders)
- `subsection(title)` - Print subsection header (with dashes)
- `info(message)` - Print informational message
- `success(message)` - Print success message (✓ prefix)
- `warning(message)` - Print warning (⚠ prefix, tracked in errors list)
- `error(message)` - Print error to stderr (✗ prefix, tracked in errors list)
- `result(key, value, unit)` - Print measurement result with optional unit

**Utilities**:
- `get_data_dir()` - Path to demonstrations/data/
- `get_output_dir()` - Path to demonstrations/data/outputs/<demo_name>/

### Data Generation (`data_generation.py`)

Functions for creating realistic synthetic signals:

**Wave Generators**:
- `generate_sine_wave(frequency, amplitude, duration, sample_rate, offset, phase)` - Pure sine wave
- `generate_square_wave(frequency, amplitude, duration, sample_rate, duty_cycle, offset)` - Square wave
- `generate_pulse_train(pulse_width, period, amplitude, duration, sample_rate, rise_time, fall_time)` - Realistic digital pulses

**Complex Signals**:
- `generate_complex_signal(fundamentals, amplitudes, duration, sample_rate, snr_db)` - Multiple frequency components
- `add_noise(trace, snr_db)` - Add white Gaussian noise with specified SNR

### Constants (`constants.py`)

SSOT for constants used across demonstrations:

**Tolerances**:
```python
TOLERANCE_STRICT = 0.01      # 1% precise measurements
TOLERANCE_NORMAL = 0.05      # 5% typical measurements
TOLERANCE_RELAXED = 0.10     # 10% noisy measurements
```

**Precision**:
```python
FLOAT_EPSILON = 1e-14        # Near-zero threshold
FLOAT_TOLERANCE = 1e-6       # Relative comparison tolerance
```

**Mathematical**:
```python
SINE_RMS_FACTOR = 1/√2       # Peak to RMS conversion
SQRT2 = √2                   # RMS to peak conversion
```

**Random**:
```python
RANDOM_SEED = 42             # Deterministic test data generation
```

### Validation (`validation.py`)

Validation helpers used in `validate()` methods:

- `validate_approximately(actual, expected, tolerance, name)` - Floating-point with tolerance
- `validate_range(value, min_val, max_val, name)` - Bounded value checking
- `validate_exists(obj, name)` - Non-None validation
- `validate_length(seq, expected_length, name)` - Sequence length checking
- `validate_type(obj, expected_type, name)` - Type validation
- `validate_results(results, expected)` - Batch validation of multiple results

### Utilities (Other Modules)

Additional utility modules in `demonstrations/common/`:

- `formatting.py` - Output formatting helpers
- `plotting.py` - Visualization utilities (if visualization is included)

---

## Best Practices for New Demonstrations

### 1. Structure

```python
"""Module docstring explaining what is demonstrated.

Demonstrates:
- oscura.capability1
- oscura.capability2

IEEE Standards: IEEE 1241-2010

Related Demos:
- path/to/related.py
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, ...
from oscura import ...

class MyDemo(BaseDemo):
    # Implementation

if __name__ == "__main__":
    demo = MyDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
```

### 2. Inherit from BaseDemo

```python
class MyDemo(BaseDemo):
    def __init__(self):
        super().__init__(
            name="my_demo",  # snake_case, matches file prefix
            description="Short description (one line)",
            capabilities=["oscura.feature1", "oscura.feature2"],
            ieee_standards=["IEEE XXXX-YYYY"] if applicable,
            related_demos=["path/to/related.py"],
        )
```

### 3. Support --data-file Argument

The `BaseDemo.execute()` method automatically handles `--data-file`:

```bash
# Run with synthetic data (default)
python demonstrations/my_demo.py

# Run with custom data
python demonstrations/my_demo.py --data-file my_data.npz
```

Implement `load_custom_data()` only if needed:

```python
def load_custom_data(self, data_file: str) -> dict:
    """Custom data loading logic if needed."""
    # BaseDemo provides default NPZ support
```

### 4. Seed Random Number Generators

```python
def generate_test_data(self) -> dict:
    import numpy as np
    from demonstrations.common.constants import RANDOM_SEED

    np.random.seed(RANDOM_SEED)  # Deterministic generation
    # ... generate data ...
```

### 5. Use Constants from constants.py

```python
from demonstrations.common.constants import (
    TOLERANCE_STRICT,
    TOLERANCE_NORMAL,
    TOLERANCE_RELAXED,
    SINE_RMS_FACTOR,
    RANDOM_SEED,
)
```

### 6. Derive Validation Values from Generation Parameters

```python
def generate_test_data(self) -> dict:
    freq_hz = 1000.0  # Save parameter
    amp_v = 1.0       # Save parameter
    trace = generate_sine_wave(frequency=freq_hz, amplitude=amp_v)
    return {"trace": trace, "freq_hz": freq_hz, "amp_v": amp_v}

def run_demonstration(self, data: dict) -> dict:
    measured_freq = frequency(data["trace"])
    return {"measured_freq": measured_freq}

def validate(self, results: dict) -> bool:
    # Use data['freq_hz'] from generate_test_data as expected value
    expected_freq = self.last_data["freq_hz"]  # Or pass via return
    return validate_approximately(
        results["measured_freq"],
        expected_freq,
        tolerance=TOLERANCE_NORMAL,
    )
```

### 7. Keep Docstrings Focused

```python
def generate_test_data(self) -> dict:
    """Generate test signals for spectral analysis.

    Generates:
    1. Pure sine: 1kHz at 1V peak
    2. With harmonics: +3rd/5th harmonics
    3. With noise: 40dB SNR

    Returns:
        Dict with sine_wave, harmonics_wave, noisy_wave
    """
```

Keep to 10-20 lines explaining what signals are generated and why.

### 8. Add to Capability Index

Edit `demonstrations/common/__init__.py` to ensure proper imports:

```python
# Re-export for easier access
from demonstrations.common.base_demo import BaseDemo
from demonstrations.common.data_generation import (
    generate_sine_wave,
    generate_square_wave,
    add_noise,
)
from demonstrations.common.validation import validate_approximately
from demonstrations.common.constants import TOLERANCE_NORMAL
```

---

## File Organization

```
demonstrations/
├── ARCHITECTURE.md                    # This file
├── README.md                          # Framework overview
├── validate_all.py                    # Test runner (112 demos)
├── capability_index.py                # API coverage analyzer
├── generate_all_data.py               # Pre-generate all test data
│
├── 00_getting_started/                # Beginner tutorials
│   ├── 00_hello_world.py
│   ├── 01_core_types.py
│   └── 02_supported_formats.py
│
├── 01_data_loading/                   # File format loaders
│   ├── 01_oscilloscopes.py
│   ├── 02_logic_analyzers.py
│   └── ...
│
├── 02_basic_analysis/                 # Common measurements
│   ├── 01_waveform_measurements.py
│   ├── 02_statistical_measurements.py
│   ├── 03_spectral_analysis.py
│   └── ...
│
├── 03_protocol_decoding/              # Protocol decoders
│   ├── 01_serial_comprehensive.py
│   ├── 02_automotive_protocols.py
│   └── ...
│
├── 04_advanced_analysis/              # Complex analysis
│   ├── 01_correlation_analysis.py
│   ├── 02_power_analysis.py
│   └── ...
│
├── 19_standards_compliance/           # IEEE standards
│   ├── 01_ieee_181.py
│   ├── 02_ieee_1241.py
│   └── ...
│
├── common/                            # Shared utilities
│   ├── __init__.py
│   ├── base_demo.py                  # Template class
│   ├── constants.py                  # SSOT constants
│   ├── data_generation.py            # Signal generators
│   ├── validation.py                 # Validation helpers
│   ├── formatting.py                 # Output formatting
│   └── plotting.py                   # Visualization
│
└── data/                              # Test data directory
    ├── outputs/                       # Generated outputs
    │   ├── hello_world/
    │   ├── spectral_analysis/
    │   └── ...
    └── README.md                      # Data directory guide
```

### Naming Conventions

- **Directories**: `NN_descriptive_name` (e.g., `00_getting_started`, `03_protocol_decoding`)
- **Demonstrations**: `NN_descriptive_name.py` (e.g., `00_hello_world.py`, `03_spectral_analysis.py`)
- **Classes**: `PascalCase + Demo` (e.g., `HelloWorldDemo`, `SpectralAnalysisDemo`)
- **Functions**: `snake_case` (standard Python)

---

## How the System Works

### Running Demonstrations

**Run all demonstrations** (112 total):
```bash
python demonstrations/validate_all.py
```

**Run single demonstration**:
```bash
python demonstrations/00_getting_started/00_hello_world.py
```

**Run with custom data**:
```bash
python demonstrations/02_basic_analysis/03_spectral_analysis.py --data-file my_data.npz
```

**Run specific section**:
```bash
python demonstrations/validate_all.py --section 02_basic_analysis
```

### Execution Flow

1. **Test Discovery** - `validate_all.py` finds all demonstration files
2. **Demo Instantiation** - Creates instance of demonstration class
3. **Execute Framework** - Calls `execute()` which:
   - Generates synthetic test data via `generate_test_data()`
   - Runs demonstration via `run_demonstration(data)`
   - Validates results via `validate(results)`
   - Returns True/False pass/fail
4. **Result Collection** - Aggregates results from all demos
5. **Report Generation** - Prints summary with pass/fail counts

### Capability Tracking

1. **AST Parsing** - `capability_index.py` parses all demonstration files
2. **Import Extraction** - Finds imports from `oscura.*` modules
3. **Call Extraction** - Identifies function calls to `oscura` APIs
4. **Coverage Analysis** - Matches capabilities to API symbols
5. **Report Generation** - Creates coverage matrix and gap analysis

---

## Key Design Principles

### Single Source of Truth (SSOT)

Constants live in one place:
- Tolerances in `demonstrations/common/constants.py`
- Test data generation parameters in individual demos
- Validation logic in `demonstrations/common/validation.py`

### Deterministic Reproducibility

- Always seed RNGs with `RANDOM_SEED`
- Generate all test data in-memory (no external files)
- Same parameters always produce same results

### Progressive Disclosure

- `00_getting_started/` - Simple, beginner-friendly demonstrations
- `01_data_loading/` through `05_domain_specific/` - Progressively complex
- `06_reverse_engineering/` and beyond - Advanced scenarios

### Self-Documenting Code

- Module docstrings explain what's demonstrated
- Generation parameters show realistic scenarios
- Validation values derive from generation parameters
- Capability lists are comprehensive and accurate

---

## Troubleshooting

### Demonstration Fails to Run

**Problem**: `ImportError: No module named 'oscura'`

**Solution**: Ensure Oscura is installed:
```bash
cd /path/to/oscura
uv sync  # Install dependencies
```

### Validation Fails

**Problem**: `✗ Amplitude: 1.95 != 2.0 (diff 0.05 > 0.02)`

**Solution**: Check tolerance levels:
```python
# If measurement is inherently noisy, use TOLERANCE_RELAXED
validate_approximately(value, expected, tolerance=TOLERANCE_RELAXED)
```

### Custom Data File Not Loading

**Problem**: `ValueError: Unsupported file format: .csv`

**Solution**: Currently only NPZ format is supported. Convert data:
```python
import numpy as np

# Convert CSV to NPZ
data = np.loadtxt("data.csv")
np.savez_compressed("data.npz", signal=data)
```

---

## Contributing New Demonstrations

1. **Create file** in appropriate category directory
2. **Inherit from BaseDemo** with required metadata
3. **Implement three methods**:
   - `generate_test_data()` - Create synthetic data
   - `run_demonstration(data)` - Run the demonstration
   - `validate(results)` - Validate correctness
4. **Use common utilities** from `demonstrations/common/`
5. **Test locally**:
   ```bash
   python demonstrations/XX_category/YY_demo.py
   ```
6. **Verify in validation suite**:
   ```bash
   python demonstrations/validate_all.py
   ```

---

## References

- **BaseDemo Class**: `demonstrations/common/base_demo.py` (348 lines)
- **Validation Helpers**: `demonstrations/common/validation.py` (150 lines)
- **Data Generation**: `demonstrations/common/data_generation.py` (194 lines)
- **Constants**: `demonstrations/common/constants.py` (42 lines)
- **Test Runner**: `demonstrations/validate_all.py` (200+ lines)
- **Capability Index**: `demonstrations/capability_index.py` (318 lines)

---

**Last Updated**: 2024
**Framework Version**: 1.0
**Total Demonstrations**: 112
**Coverage**: 92%+ of Oscura API
