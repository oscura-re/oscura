# Example Validation

## Overview

This document describes the automated validation system for examples and demonstrations
in the Oscura repository.

## Validation Infrastructure

### Test Suite

All examples are validated through an automated pytest suite:

- **Location**: `tests/integration/test_examples.py`
- **Execution**: `./scripts/test.sh tests/integration/test_examples.py`
- **Coverage**: 175+ examples across demonstrations/, examples/, and demos/

### Validation Scripts

#### Quick Validation

```bash
./.claude/validate_examples.sh
```

Runs all examples with colored output and generates logs in `.claude/analysis/example_validation/`.

#### Detailed Validation

```bash
python3 .claude/scripts/validate_examples_detailed.py
```

Generates comprehensive report with:

- Failure categorization
- Error analysis
- Recommendations for fixes
- Output saved to `.claude/analysis/example_validation_report.txt`

## Example Categories

### Demonstrations (`demonstrations/`)

**Purpose**: Educational examples organized by feature category

**Categories**:

- `00_getting_started/` - Basic usage (3 examples)
- `01_data_loading/` - File format loaders (10 examples)
- `02_basic_analysis/` - Measurements and analysis (7 examples)
- `03_protocol_decoding/` - Protocol analyzers (6 examples)
- `04_advanced_analysis/` - Advanced techniques (9 examples)
- `05_domain_specific/` - Industry-specific workflows (4 examples)
- `06_reverse_engineering/` - RE techniques (10 examples)
- `07_advanced_api/` - API composition (8 examples)
- `08_extensibility/` - Plugin system (6 examples)
- `09_batch_processing/` - Parallel workflows (4 examples)
- `10_sessions/` - Session management (5 examples)
- `11_integration/` - External integrations (5 examples)
- `12_quality_tools/` - Quality assessment (4 examples)
- `13_guidance/` - Smart recommendations (3 examples)
- `14_exploratory/` - Unknown signal analysis (5 examples)
- `15_export_visualization/` - Export formats (6 examples)
- `16_complete_workflows/` - End-to-end examples (6 examples)
- `17_signal_generation/` - Signal synthesis (3 examples)
- `18_comparison_testing/` - Golden reference testing (4 examples)
- `19_standards_compliance/` - IEEE standards (4 examples)

### Demos (`demos/`)

**Purpose**: Working demonstrations with full data generation

**Structure**: Each demo includes:

- Main demo script
- Data generation script (`generate_demo_data.py`)
- Validation helpers

### Examples (`examples/`)

**Purpose**: Specific feature showcases

**Contents**:

- Automotive workflows
- ML integration
- Side-channel analysis
- Web dashboard
- Export examples

## Skipping Examples

### Marker System

Examples that require external hardware or services can be skipped:

```python
"""Example: Hardware Integration.

# SKIP_VALIDATION: Requires hardware device

This example requires a connected oscilloscope.
"""
```

### Requirements Marker

Examples with optional dependencies:

```python
"""Example: Advanced Analysis.

REQUIRES: scikit-learn, tensorflow

This example demonstrates ML-based analysis.
"""
```

## Failure Categories

The validation system automatically categorizes failures:

- **missing_module**: ModuleNotFoundError (dependency not installed)
- **import_error**: ImportError (API changes)
- **missing_file**: FileNotFoundError (test data missing)
- **timeout**: Execution exceeded 60 seconds
- **syntax_error**: Python syntax error
- **api_change**: AttributeError (API changed)
- **deprecated_api**: DeprecationWarning
- **runtime_error**: Other runtime errors

## Fixing Failed Examples

### Missing Dependencies

**Issue**: Example imports optional dependency not in pyproject.toml

**Solutions**:

1. **Add dependency** to `pyproject.toml` if feature is supported
2. **Add graceful degradation** to example:

```python
try:
    from external_lib import Feature
except ImportError:
    print("⚠ external_lib not installed. Install with:")
    print("   uv pip install 'external_lib>=1.0.0'")
    sys.exit(0)  # Exit gracefully
```

1. **Mark for manual testing** with `# SKIP_VALIDATION` marker

### API Changes

**Issue**: Example uses deprecated or changed API

**Solution**: Update to current API:

```python
# Old (deprecated)
trace = osc.WaveformTrace(data, metadata)

# New (current)
from oscura.core.types import WaveformTrace, TraceMetadata
metadata = TraceMetadata(sample_rate=1e6, unit="V")
trace = WaveformTrace(data=data, metadata=metadata)
```

### Missing Test Data

**Issue**: Example expects data file that doesn't exist

**Solutions**:

1. **Generate synthetic data** in example:

```python
def generate_test_data():
    """Generate synthetic test data inline."""
    import numpy as np
    t = np.linspace(0, 1, 1000)
    return np.sin(2 * np.pi * 10 * t)
```

1. **Use common data generators** from `demonstrations/common/`:

```python
from demonstrations.common import generate_sine_wave

data = generate_sine_wave(frequency=1000, sample_rate=100e3, duration=0.01)
```

### Timeout Issues

**Issue**: Example takes >60 seconds to execute

**Solutions**:

1. **Reduce data size** for example:

```python
# Too large (timeout)
n_samples = 100_000_000

# Better for example
n_samples = 10_000  # Sufficient for demonstration
```

1. **Optimize algorithms** to reduce computation
2. **Mark as long-running** with appropriate timeout in test

## Best Practices

### Example Structure

All examples should follow this structure:

```python
#!/usr/bin/env python3
"""Brief description of what this example demonstrates.

This example shows...

Example:
    python examples/my_example.py

Output:
    Expected output description
"""

import sys
from pathlib import Path

import numpy as np

from oscura import load_waveform
from oscura.analyzers import calculate_frequency


def main() -> None:
    """Run example."""
    # 1. Generate or load test data
    print("Loading test data...")
    data = generate_test_signal()

    # 2. Perform analysis
    print("Analyzing signal...")
    result = calculate_frequency(data, sample_rate=1e6)

    # 3. Display results
    print(f"\nFrequency: {result:.2f} Hz")
    print("✓ Analysis complete")


def generate_test_signal() -> np.ndarray:
    """Generate synthetic test signal.

    Returns:
        1 kHz sine wave (1000 samples)
    """
    t = np.linspace(0, 0.001, 1000)
    return np.sin(2 * np.pi * 1000 * t)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"✗ Example failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    else:
        print("✓ Example completed successfully")
```

### Key Requirements

1. **Self-contained**: Generate test data inline or use common utilities
2. **Fast**: Execute in <30 seconds
3. **Clear output**: Print progress and results
4. **Error handling**: Catch exceptions and provide clear messages
5. **Success indicator**: Print "✓ Example completed successfully"
6. **Docstring**: Explain purpose, usage, and expected output

### Data Generation

**Preferred approaches** (best to worst):

1. **Inline synthetic data**: Generate with numpy in example
2. **Common utilities**: Use `demonstrations/common/data_generation.py`
3. **Committed test data**: Small files (<100KB) in repository
4. **Generated test data**: Run `generate_demo_data.py` before example
5. **External data**: Mark with `# SKIP_VALIDATION`

## Continuous Integration

### Pre-commit

Examples are NOT validated on every commit (too slow).

### Pull Request CI

Examples ARE validated for:

- PRs affecting `src/oscura/` (API changes)
- PRs affecting examples themselves
- Nightly full validation runs

### Nightly Tests

Full example validation runs nightly:

- All 175+ examples executed
- Report generated
- Failures trigger issue creation

## Metrics

### Target: 100% Success Rate

- **Current baseline**: Measured during initial validation
- **Goal**: Zero broken examples
- **Tolerance**: <5% failure rate acceptable if marked SKIP_VALIDATION

### Performance Targets

- **Individual example**: <30 seconds execution
- **Full suite**: <30 minutes total execution
- **Fast examples**: <10 seconds (80% of examples)

## Maintenance

### When Adding New Example

1. Add example file with proper structure
2. Run validation: `uv run pytest tests/integration/test_examples.py -k "my_example"`
3. Ensure example passes
4. Add to appropriate category directory
5. Update this documentation if new pattern introduced

### When Changing API

1. Search for affected examples: `grep -r "old_api_name" examples/ demonstrations/ demos/`
2. Update all affected examples
3. Run full validation suite
4. Verify all examples still pass

### Quarterly Review

Every quarter:

1. Run full validation report
2. Review skipped examples (can they be un-skipped?)
3. Update to use latest API patterns
4. Optimize slow examples
5. Update documentation

## See Also

- `tests/integration/test_examples.py` - Test implementation
- `.claude/validate_examples.sh` - Quick validation script
- `.claude/scripts/validate_examples_detailed.py` - Detailed validation
- `demonstrations/README.md` - Demonstration structure
- `CONTRIBUTING.md` - Contributing guidelines
