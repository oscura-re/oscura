# Comparison and Limits API Reference

> **Version**: 0.6.0
> **Last Updated**: 2026-01-25

## Overview

Oscura provides comprehensive waveform comparison and limit testing capabilities including trace comparison, golden reference testing, specification limit checking, and mask-based pass/fail testing. These features are essential for validation, regression testing, and quality assurance of signal integrity.

## Quick Start

```python
import oscura as osc

# All comparison functions available at top-level
# Or import explicitly:
# from oscura.utils.comparison import (
#     compare_traces, similarity_score,
#     create_golden, compare_to_golden,
#     create_limit_spec, check_limits, margin_analysis,
#     eye_mask, mask_test
# )

# Load waveforms
measured = osc.load("measured.wfm")
reference = osc.load("reference.wfm")

# Basic comparison
result = osc.compare_traces(measured, reference, tolerance=0.01)
print(f"Match: {result.match}, Similarity: {result.similarity:.1%}")

# Golden reference testing
golden = osc.create_golden(reference, tolerance_pct=5)
test_result = osc.compare_to_golden(measured, golden)

# Limit testing
spec = osc.create_limit_spec(upper=3.3, lower=0.0, guardband_pct=10)
limit_result = osc.check_limits(measured, spec)

# Eye diagram mask testing
mask = osc.eye_mask(eye_width=0.5, eye_height=0.4)
mask_result = osc.mask_test(eye_trace, mask)
```

## Trace Comparison

### `compare_traces()`

Comprehensive comparison of two waveforms with difference analysis, correlation, and match determination.

```python
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import compare_traces

result = compare_traces(
    trace1,              # First trace (typically measured)
    trace2,              # Second trace (typically reference)
    tolerance=0.01,      # Absolute tolerance for matching
    tolerance_pct=None,  # Percentage tolerance (0-100)
    method="absolute",   # "absolute", "relative", or "statistical"
    include_difference=True
)

# Access results
if result.match:
    print(f"✓ Traces match! Similarity: {result.similarity:.1%}")
else:
    print(f"✗ Traces differ! Max difference: {result.max_difference:.4f}")
    print(f"  Violations: {result.num_violations}")

print(f"Correlation: {result.correlation:.3f}")
print(f"RMS difference: {result.rms_difference:.6f}")

# Optional difference trace
if result.difference_trace:
    osc.plot(result.difference_trace)
```

**Parameters:**

| Parameter            | Type                                      | Default      | Description                        |
| -------------------- | ----------------------------------------- | ------------ | ---------------------------------- |
| `trace1`             | `WaveformTrace`                           | required     | First trace (typically measured)   |
| `trace2`             | `WaveformTrace`                           | required     | Second trace (typically reference) |
| `tolerance`          | `float \|None`                            | `None`       | Absolute tolerance for matching    |
| `tolerance_pct`      | `float \|None`                            | `None`       | Percentage tolerance (0-100)       |
| `method`             | `"absolute" \|"relative" \|"statistical"` | `"absolute"` | Comparison method                  |
| `include_difference` | `bool`                                    | `True`       | Include difference trace in result |

**Returns:** `ComparisonResult`

**ComparisonResult Attributes:**

- `match: bool` - True if traces are considered matching
- `similarity: float` - Similarity score (0.0 to 1.0)
- `max_difference: float` - Maximum absolute difference
- `rms_difference: float` - RMS of the difference
- `correlation: float` - Correlation coefficient
- `difference_trace: WaveformTrace | None` - Difference waveform
- `violations: NDArray[np.int64] | None` - Indices where difference exceeds threshold
- `statistics: dict | None` - Additional comparison statistics

### `difference()`

Compute element-wise difference between two traces.

```python
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import difference

diff = difference(
    trace1,                  # First trace
    trace2,                  # Second trace
    normalize=False,         # Normalize to percentage
    channel_name=None        # Name for result trace
)

# Analyze difference
max_error = np.max(np.abs(diff.data))
print(f"Maximum error: {max_error:.6f}")

# Plot difference
osc.plot(diff)
```

**Example - Normalized difference:**

```python
# Express difference as percentage of reference range
diff_pct = difference(measured, reference, normalize=True)
print(f"Max error: {np.max(np.abs(diff_pct.data)):.2f}%")
```

### `correlation()`

Compute cross-correlation between two traces for pattern matching and delay detection.

```python
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import correlation

lags, corr_values = correlation(
    trace1,
    trace2,
    mode="same",        # "full", "same", or "valid"
    normalize=True      # Normalize to correlation coefficient
)

# Find time delay
delay_samples = lags[np.argmax(corr_values)]
delay_time = delay_samples / trace1.metadata.sample_rate
print(f"Delay: {delay_time*1e9:.2f} ns ({delay_samples} samples)")

# Plot correlation
import matplotlib.pyplot as plt
plt.plot(lags, corr_values)
plt.xlabel("Lag (samples)")
plt.ylabel("Correlation")
plt.title("Cross-Correlation")
plt.show()
```

**Parameters:**

| Parameter   | Type                        | Default  | Description                                    |
| ----------- | --------------------------- | -------- | ---------------------------------------------- |
| `trace1`    | `WaveformTrace`             | required | First trace                                    |
| `trace2`    | `WaveformTrace`             | required | Second trace                                   |
| `mode`      | `"full" \|"same" \|"valid"` | `"same"` | Correlation mode                               |
| `normalize` | `bool`                      | `True`   | Normalize to correlation coefficient (-1 to 1) |

**Returns:** `tuple[NDArray[np.float64], NDArray[np.float64]]` - (lags, correlation_values)

### `similarity_score()`

Compute similarity score between two traces using various metrics.

```python
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import similarity_score

# Correlation-based similarity (default)
score = similarity_score(trace1, trace2, method="correlation")

# Alternative methods
score_rms = similarity_score(trace1, trace2, method="rms")
score_mse = similarity_score(trace1, trace2, method="mse")
score_cosine = similarity_score(trace1, trace2, method="cosine")

# With normalization options
score = similarity_score(
    trace1,
    trace2,
    method="correlation",
    normalize_amplitude=True,  # Normalize amplitude
    normalize_offset=True      # Remove DC offset
)

if score > 0.95:
    print("✓ Traces are highly similar")
elif score > 0.80:
    print("⚠ Traces are moderately similar")
else:
    print("✗ Traces are significantly different")
```

**Parameters:**

| Parameter             | Type                                       | Default         | Description                           |
| --------------------- | ------------------------------------------ | --------------- | ------------------------------------- |
| `trace1`              | `WaveformTrace`                            | required        | First trace                           |
| `trace2`              | `WaveformTrace`                            | required        | Second trace                          |
| `method`              | `"correlation" \|"rms" \|"mse" \|"cosine"` | `"correlation"` | Similarity metric                     |
| `normalize_amplitude` | `bool`                                     | `True`          | Normalize amplitude before comparison |
| `normalize_offset`    | `bool`                                     | `True`          | Remove DC offset before comparison    |

**Returns:** `float` - Similarity score from 0.0 (completely different) to 1.0 (identical)

**Methods:**

- `correlation` - Pearson correlation coefficient (robust to amplitude/offset)
- `rms` - 1 - normalized RMS difference
- `mse` - 1 - normalized mean squared error
- `cosine` - Cosine similarity (angle between vectors)

## Golden Reference Testing

Golden reference testing compares measured waveforms against a known-good reference with tolerance bounds for automated pass/fail testing.

### `GoldenReference`

A golden reference waveform with tolerance bounds for pass/fail testing.

**Attributes:**

- `data: NDArray[np.float64]` - Reference waveform data
- `sample_rate: float` - Sample rate in Hz
- `upper_bound: NDArray[np.float64]` - Upper tolerance bound
- `lower_bound: NDArray[np.float64]` - Lower tolerance bound
- `tolerance: float` - Tolerance used to create bounds
- `tolerance_type: Literal["absolute", "percentage", "sigma"]` - How tolerance was applied
- `name: str` - Reference name
- `description: str` - Optional description
- `created: datetime` - Creation timestamp
- `metadata: dict[str, Any]` - Additional metadata

**Properties:**

- `num_samples: int` - Number of samples
- `duration: float` - Duration in seconds

**Methods:**

- `save(path)` - Save to JSON file
- `load(path)` - Load from JSON file (class method)
- `to_dict()` - Convert to dictionary
- `from_dict(data)` - Create from dictionary (class method)

### `create_golden()`

Create a golden reference from a trace with tolerance bounds.

```python
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import create_golden

# Create with percentage tolerance
golden = create_golden(
    reference_trace,
    tolerance_pct=5.0,           # 5% of signal range
    name="power_rail_3v3",
    description="3.3V power rail under load"
)

# Create with absolute tolerance
golden = create_golden(
    reference_trace,
    tolerance=0.05,              # 50mV absolute
    name="reference_voltage"
)

# Create with sigma tolerance (statistical)
golden = create_golden(
    reference_trace,
    tolerance_sigma=3.0,         # 3 standard deviations
    name="noisy_signal"
)

# Save for later use
golden.save("golden_references/power_3v3.json")

# Load existing golden reference
golden = GoldenReference.load("golden_references/power_3v3.json")
```

**Parameters:**

| Parameter         | Type            | Default    | Description                                 |
| ----------------- | --------------- | ---------- | ------------------------------------------- |
| `trace`           | `WaveformTrace` | required   | Reference waveform trace                    |
| `tolerance`       | `float \|None`  | `None`     | Absolute tolerance value                    |
| `tolerance_pct`   | `float \|None`  | `None`     | Percentage tolerance (0-100)                |
| `tolerance_sigma` | `float \|None`  | `None`     | Tolerance as multiple of standard deviation |
| `name`            | `str`           | `"golden"` | Name for the reference                      |
| `description`     | `str`           | `""`       | Optional description                        |

**Returns:** `GoldenReference`

**Note:** Exactly one of `tolerance`, `tolerance_pct`, or `tolerance_sigma` should be specified. If none are provided, defaults to 1% of signal range.

### `compare_to_golden()`

Compare a measured trace to a golden reference for pass/fail testing.

```python
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import compare_to_golden

result = compare_to_golden(
    measured_trace,
    golden,
    align=True,         # Auto-align traces by correlation
    interpolate=True    # Interpolate if lengths differ
)

# Pass/fail determination
if result.passed:
    print("✓ PASS: Trace within tolerance")
    print(f"  Margin: {result.margin_percentage:.1f}%")
else:
    print("✗ FAIL: Trace outside tolerance")
    print(f"  Violations: {result.num_violations}")
    print(f"  Violation rate: {result.violation_rate:.1%}")
    print(f"  Max deviation: {result.max_deviation:.6f}")

# Detailed statistics
print(f"RMS deviation: {result.rms_deviation:.6f}")
print(f"Upper violations: {len(result.upper_violations) if result.upper_violations is not None else 0}")
print(f"Lower violations: {len(result.lower_violations) if result.lower_violations is not None else 0}")
```

**Parameters:**

| Parameter     | Type              | Default  | Description                                  |
| ------------- | ----------------- | -------- | -------------------------------------------- |
| `trace`       | `WaveformTrace`   | required | Measured trace to compare                    |
| `golden`      | `GoldenReference` | required | Golden reference to compare against          |
| `align`       | `bool`            | `True`   | Attempt to align traces by cross-correlation |
| `interpolate` | `bool`            | `True`   | Interpolate if sample counts differ          |

**Returns:** `GoldenComparisonResult`

**GoldenComparisonResult Attributes:**

- `passed: bool` - True if measured waveform is within tolerance
- `num_violations: int` - Number of samples outside tolerance
- `violation_rate: float` - Fraction of samples outside tolerance
- `max_deviation: float` - Maximum deviation from reference
- `rms_deviation: float` - RMS deviation from reference
- `upper_violations: NDArray[np.int64] | None` - Indices exceeding upper bound
- `lower_violations: NDArray[np.int64] | None` - Indices below lower bound
- `margin: float | None` - Minimum margin to tolerance bound
- `margin_percentage: float | None` - Margin as percentage of tolerance
- `statistics: dict[str, Any]` - Additional comparison statistics

**Example - Batch testing:**

```python
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import batch_compare_to_golden

# Test multiple units
results = batch_compare_to_golden(measured_traces, golden)

# Compute pass rate
pass_rate = sum(r.passed for r in results) / len(results)
print(f"Pass rate: {pass_rate:.1%}")

# Find worst case
worst = min(results, key=lambda r: r.margin_percentage or 0)
print(f"Worst case margin: {worst.margin_percentage:.1f}%")
```

**Example - Creating golden from multiple samples:**

```python
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import golden_from_average

# Average multiple good samples
golden = golden_from_average(
    good_samples,
    tolerance_sigma=3.0,
    name="averaged_reference"
)
```

## Specification Limit Testing

Test waveforms against specification limits with guardband support and margin analysis.

### `LimitSpec`

Specification limit definition with upper/lower bounds and guardbands.

**Attributes:**

- `upper: float | None` - Upper limit value
- `lower: float | None` - Lower limit value
- `upper_guardband: float` - Guardband below upper limit (margin)
- `lower_guardband: float` - Guardband above lower limit (margin)
- `name: str` - Name of the specification
- `unit: str` - Unit of measurement
- `mode: Literal["absolute", "relative"]` - Limit mode

**Validation:** At least one of `upper` or `lower` must be specified, and `upper >= lower`.

### `create_limit_spec()`

Create a limit specification with convenient notation support.

```python
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import create_limit_spec

# Simple upper/lower limits
spec = create_limit_spec(
    upper=3.3,
    lower=2.7,
    name="3V3_rail",
    unit="V"
)

# Center +/- tolerance notation
spec = create_limit_spec(
    center=3.0,
    tolerance=0.3,           # 3.0 ± 0.3V
    name="3V0_rail",
    unit="V"
)

# Center +/- percentage notation
spec = create_limit_spec(
    center=3.3,
    tolerance_pct=5,         # 3.3V ± 5% = 3.135V to 3.465V
    name="3V3_rail_5pct",
    unit="V"
)

# With guardband (test margin)
spec = create_limit_spec(
    upper=3.3,
    lower=2.7,
    guardband_pct=10,        # 10% guardband (warning zone)
    name="3V3_with_guardband",
    unit="V"
)
```

**Parameters:**

| Parameter       | Type           | Default  | Description                            |
| --------------- | -------------- | -------- | -------------------------------------- |
| `upper`         | `float \|None` | `None`   | Upper limit value                      |
| `lower`         | `float \|None` | `None`   | Lower limit value                      |
| `center`        | `float \|None` | `None`   | Center value (used with tolerance)     |
| `tolerance`     | `float \|None` | `None`   | Absolute tolerance (+/- from center)   |
| `tolerance_pct` | `float \|None` | `None`   | Percentage tolerance (+/- % of center) |
| `guardband_pct` | `float`        | `0.0`    | Guardband as percentage of limit range |
| `name`          | `str`          | `"spec"` | Specification name                     |
| `unit`          | `str`          | `""`     | Unit of measurement                    |

**Returns:** `LimitSpec`

### `check_limits()`

Check if trace data is within specification limits.

```python
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import check_limits

# Simple check with direct limits
result = check_limits(
    trace,
    upper=3.465,
    lower=3.135
)

# Or use LimitSpec
spec = create_limit_spec(center=3.3, tolerance_pct=5)
result = check_limits(trace, limits=spec)

# Analyze results
if result.passed:
    print("✓ PASS: All samples within limits")
    print(f"  Max value: {result.max_value:.4f}")
    print(f"  Min value: {result.min_value:.4f}")
    print(f"  Upper margin: {result.upper_margin:.4f}")
    print(f"  Lower margin: {result.lower_margin:.4f}")

    if result.within_guardband:
        print("  ⚠ WARNING: Within guardband zone")
else:
    print("✗ FAIL: Samples outside limits")
    print(f"  Violations: {result.num_violations}")
    print(f"  Violation rate: {result.violation_rate:.1%}")

    if result.upper_violations is not None:
        print(f"  Upper limit violations: {len(result.upper_violations)}")
    if result.lower_violations is not None:
        print(f"  Lower limit violations: {len(result.lower_violations)}")
```

**Parameters:**

| Parameter   | Type                      | Default  | Description                            |
| ----------- | ------------------------- | -------- | -------------------------------------- |
| `trace`     | `WaveformTrace \|NDArray` | required | Input trace or data array              |
| `limits`    | `LimitSpec \|None`        | `None`   | LimitSpec defining the limits          |
| `upper`     | `float \|None`            | `None`   | Upper limit (alternative to LimitSpec) |
| `lower`     | `float \|None`            | `None`   | Lower limit (alternative to LimitSpec) |
| `reference` | `float \|None`            | `None`   | Reference value for relative limits    |

**Returns:** `LimitTestResult`

**LimitTestResult Attributes:**

- `passed: bool` - True if all samples are within limits
- `num_violations: int` - Number of samples violating limits
- `violation_rate: float` - Fraction of samples violating limits
- `upper_violations: NDArray[np.int64] | None` - Indices of samples exceeding upper limit
- `lower_violations: NDArray[np.int64] | None` - Indices of samples below lower limit
- `max_value: float` - Maximum value in data
- `min_value: float` - Minimum value in data
- `upper_margin: float | None` - Margin to upper limit (positive = within)
- `lower_margin: float | None` - Margin to lower limit (positive = within)
- `margin_percentage: float | None` - Smallest margin as percentage of limit range
- `within_guardband: bool` - True if within guardband but outside tight limits

### `margin_analysis()`

Analyze margins to specification limits for design validation.

```python
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import margin_analysis

spec = create_limit_spec(upper=3.465, lower=3.135)
margins = margin_analysis(
    trace,
    spec,
    warning_threshold_pct=20.0  # Warn if margin < 20%
)

# Check margin status
if margins.margin_status == "pass":
    print(f"✓ Good margin: {margins.margin_percentage:.1f}%")
elif margins.margin_status == "warning":
    print(f"⚠ Low margin: {margins.margin_percentage:.1f}%")
    print(f"  Critical limit: {margins.critical_limit}")
else:  # "fail"
    print(f"✗ Negative margin: {margins.margin_percentage:.1f}%")

# Detailed margins
print(f"Upper margin: {margins.upper_margin:.4f}")
print(f"Lower margin: {margins.lower_margin:.4f}")
print(f"Minimum margin: {margins.min_margin:.4f}")
```

**Parameters:**

| Parameter               | Type                      | Default  | Description                            |
| ----------------------- | ------------------------- | -------- | -------------------------------------- |
| `trace`                 | `WaveformTrace \|NDArray` | required | Input trace or data array              |
| `limits`                | `LimitSpec`               | required | LimitSpec defining the limits          |
| `warning_threshold_pct` | `float`                   | `20.0`   | Threshold for margin warning (percent) |

**Returns:** `MarginAnalysis`

**MarginAnalysis Attributes:**

- `upper_margin: float | None` - Margin to upper limit
- `lower_margin: float | None` - Margin to lower limit
- `min_margin: float` - Smallest margin (most critical)
- `margin_percentage: float` - Margin as percentage of limit range
- `critical_limit: Literal["upper", "lower", "both", "none"]` - Which limit has smallest margin
- `warning: bool` - True if margin is below warning threshold
- `margin_status: Literal["pass", "warning", "fail"]` - Overall margin status

## Mask Testing

Mask-based pass/fail testing for waveforms, including eye diagram masks and custom polygon masks.

### `Mask`

A mask definition consisting of one or more polygon regions.

**Attributes:**

- `regions: list[MaskRegion]` - List of mask regions
- `name: str` - Name of the mask
- `x_unit: str` - Unit for X axis (e.g., "UI", "ns", "samples")
- `y_unit: str` - Unit for Y axis (e.g., "V", "mV", "normalized")
- `description: str` - Optional description

**Methods:**

- `add_region(vertices, region_type="violation", name="")` - Add a region to the mask

**MaskRegion:**

- `vertices: list[tuple[float, float]]` - Polygon vertices (x, y)
- `region_type: Literal["violation", "boundary"]` - Region type
  - `"violation"` - Data must avoid this region (fail if inside)
  - `"boundary"` - Data must stay within this region (fail if outside)
- `name: str` - Optional region name

### `create_mask()`

Create a custom mask from region definitions.

```python
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import create_mask

# Define custom mask regions
mask = create_mask(
    regions=[
        {
            "vertices": [
                (0.0, 0.5), (0.5, 0.5),
                (0.5, -0.5), (0.0, -0.5)
            ],
            "type": "violation",
            "name": "center_violation"
        },
        {
            "vertices": [
                (-1.0, 1.0), (1.0, 1.0),
                (1.0, -1.0), (-1.0, -1.0)
            ],
            "type": "boundary",
            "name": "outer_boundary"
        }
    ],
    name="custom_mask",
    x_unit="UI",
    y_unit="V"
)

# Or build incrementally
mask = Mask(name="incremental", x_unit="ns", y_unit="mV")
mask.add_region(
    vertices=[(0, 100), (10, 100), (10, -100), (0, -100)],
    region_type="violation",
    name="forbidden_zone"
)
```

**Parameters:**

| Parameter | Type         | Default         | Description                 |
| --------- | ------------ | --------------- | --------------------------- |
| `regions` | `list[dict]` | required        | List of region dictionaries |
| `name`    | `str`        | `"custom_mask"` | Mask name                   |
| `x_unit`  | `str`        | `"samples"`     | X axis unit                 |
| `y_unit`  | `str`        | `"V"`           | Y axis unit                 |

**Returns:** `Mask`

### `eye_mask()`

Create a standard eye diagram mask for high-speed serial testing.

```python
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import eye_mask

# Standard eye mask (50% width, 40% height)
mask = eye_mask(
    eye_width=0.5,           # 50% of UI
    eye_height=0.4,          # 40% of amplitude
    center_height=0.3,       # Height of center violation region
    x_margin=0.1,            # X margin for boundary
    y_margin=0.1,            # Y margin for boundary
    unit_interval=1.0,       # Duration of UI
    amplitude=1.0            # Signal amplitude
)

# Custom eye mask for specific standard
mask = eye_mask(
    eye_width=0.65,          # More open eye
    eye_height=0.50,
    center_height=0.35
)
```

**Parameters:**

| Parameter       | Type    | Default | Description                                   |
| --------------- | ------- | ------- | --------------------------------------------- |
| `eye_width`     | `float` | `0.5`   | Width of eye opening (fraction of UI)         |
| `eye_height`    | `float` | `0.4`   | Height of eye opening (fraction of amplitude) |
| `center_height` | `float` | `0.3`   | Height of center violation region             |
| `x_margin`      | `float` | `0.1`   | X margin for boundary (fraction of UI)        |
| `y_margin`      | `float` | `0.1`   | Y margin for boundary (fraction of amplitude) |
| `unit_interval` | `float` | `1.0`   | Duration of unit interval                     |
| `amplitude`     | `float` | `1.0`   | Signal amplitude                              |

**Returns:** `Mask` - Eye diagram mask with hexagonal center violation region and top/bottom violation regions

### `mask_test()`

Test a waveform against a mask for pass/fail determination.

```python
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import mask_test, eye_mask

# Create mask
mask = eye_mask(eye_width=0.5, eye_height=0.4)

# Test waveform
result = mask_test(
    eye_trace,
    mask,
    x_data=None,          # Optional X coordinates
    normalize=True,       # Normalize Y to [-1, 1]
    sample_rate=None      # Optional sample rate override
)

# Analyze results
if result.passed:
    print("✓ PASS: No mask violations")
    if result.margin:
        print(f"  Margin: {result.margin:.6f}")
else:
    print("✗ FAIL: Mask violations detected")
    print(f"  Violations: {result.num_violations}")
    print(f"  Violation rate: {result.violation_rate:.1%}")

    # Violations by region
    for region, count in result.violations_by_region.items():
        if count > 0:
            print(f"  {region}: {count} violations")

# Plot violation points
if result.violation_points:
    import matplotlib.pyplot as plt
    x_viol, y_viol = zip(*result.violation_points)
    plt.scatter(x_viol, y_viol, c='red', s=1, alpha=0.5)
    plt.title("Mask Violations")
    plt.show()
```

**Parameters:**

| Parameter     | Type             | Default  | Description                       |
| ------------- | ---------------- | -------- | --------------------------------- |
| `trace`       | `WaveformTrace`  | required | Input waveform trace              |
| `mask`        | `Mask`           | required | Mask to test against              |
| `x_data`      | `NDArray \|None` | `None`   | X coordinates for each sample     |
| `normalize`   | `bool`           | `True`   | Normalize Y data to [-1, 1] range |
| `sample_rate` | `float \|None`   | `None`   | Sample rate override              |

**Returns:** `MaskTestResult`

**MaskTestResult Attributes:**

- `passed: bool` - True if all samples pass the mask test
- `num_violations: int` - Number of samples violating the mask
- `violation_rate: float` - Fraction of samples violating the mask
- `violation_points: list[tuple[float, float]]` - List of (x, y) coordinates that violated
- `violations_by_region: dict[str, int]` - Count of violations per region
- `margin: float | None` - Estimated margin to mask boundary

## Practical Examples

### Example 1: Comparing Two Waveforms

```python
import oscura as osc
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import compare_traces, difference, similarity_score
import matplotlib.pyplot as plt

# Load waveforms
measured = osc.load("production_unit_123.wfm")
reference = osc.load("golden_reference.wfm")

# Quick similarity check
sim = similarity_score(measured, reference)
print(f"Similarity: {sim:.1%}")

if sim < 0.90:
    print("⚠ Waveforms differ significantly, performing detailed analysis...")

    # Comprehensive comparison
    result = compare_traces(
        measured,
        reference,
        tolerance_pct=2.0,  # 2% tolerance
        method="absolute"
    )

    print(f"\nComparison Results:")
    print(f"  Match: {result.match}")
    print(f"  Max difference: {result.max_difference:.6f}")
    print(f"  RMS difference: {result.rms_difference:.6f}")
    print(f"  Correlation: {result.correlation:.3f}")
    print(f"  Violations: {result.num_violations}")

    # Plot difference
    if result.difference_trace:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        # Original traces
        ax1.plot(measured.data, label="Measured", alpha=0.7)
        ax1.plot(reference.data, label="Reference", alpha=0.7)
        ax1.set_ylabel("Amplitude (V)")
        ax1.legend()
        ax1.set_title("Original Waveforms")

        # Difference
        ax2.plot(result.difference_trace.data, color='red')
        ax2.set_ylabel("Difference (V)")
        ax2.set_xlabel("Sample")
        ax2.set_title("Difference (Measured - Reference)")
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.show()
```

### Example 2: Golden Reference Testing

```python
import oscura as osc
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import create_golden, compare_to_golden, GoldenReference

# === Setup: Create golden reference from good sample ===
reference_trace = osc.load("validated_good_sample.wfm")

# Create golden with 3% tolerance
golden = create_golden(
    reference_trace,
    tolerance_pct=3.0,
    name="power_supply_3v3",
    description="3.3V power supply under 500mA load"
)

# Save for production testing
golden.save("test_data/golden_power_3v3.json")

# === Production testing: Load and test against golden ===
golden = GoldenReference.load("test_data/golden_power_3v3.json")

# Test multiple units
test_results = []
unit_ids = ["UNIT_001", "UNIT_002", "UNIT_003", "UNIT_004", "UNIT_005"]

for unit_id in unit_ids:
    measured = osc.load(f"production/{unit_id}.wfm")

    result = compare_to_golden(
        measured,
        golden,
        align=True,
        interpolate=True
    )

    test_results.append({
        'unit_id': unit_id,
        'passed': result.passed,
        'margin_pct': result.margin_percentage,
        'violations': result.num_violations,
        'max_deviation': result.max_deviation
    })

    # Print result
    status = "✓ PASS" if result.passed else "✗ FAIL"
    print(f"{unit_id}: {status} (margin: {result.margin_percentage:.1f}%)")

# Summary
pass_rate = sum(1 for r in test_results if r['passed']) / len(test_results)
print(f"\nPass rate: {pass_rate:.1%}")

# Find units needing attention
for r in test_results:
    if not r['passed']:
        print(f"  {r['unit_id']}: {r['violations']} violations, "
              f"max deviation {r['max_deviation']:.6f}")
    elif r['margin_pct'] < 20:
        print(f"  {r['unit_id']}: ⚠ Low margin ({r['margin_pct']:.1f}%)")
```

### Example 3: Specification Limit Testing

```python
import oscura as osc
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import create_limit_spec, check_limits, margin_analysis
import numpy as np

# Load power rail trace
power_rail = osc.load("power_rail_3v3.wfm")

# Define specification (3.3V ± 5% with 10% guardband)
spec = create_limit_spec(
    center=3.3,
    tolerance_pct=5,      # 3.135V to 3.465V
    guardband_pct=10,     # Additional 10% margin zone
    name="3V3_RAIL_SPEC",
    unit="V"
)

# Check limits
result = check_limits(power_rail, limits=spec)

print(f"Specification Limit Test: {spec.name}")
print(f"  Limits: {spec.lower:.3f}V to {spec.upper:.3f}V")
print(f"  Measured range: {result.min_value:.3f}V to {result.max_value:.3f}V")

if result.passed:
    print("  Status: ✓ PASS")

    if result.within_guardband:
        print("  ⚠ WARNING: Within guardband zone (design margin low)")
    else:
        print("  ✓ Good design margin")

    print(f"  Upper margin: {result.upper_margin*1000:.1f} mV")
    print(f"  Lower margin: {result.lower_margin*1000:.1f} mV")

    # Detailed margin analysis
    margins = margin_analysis(power_rail, spec, warning_threshold_pct=20)

    print(f"\nMargin Analysis:")
    print(f"  Minimum margin: {margins.min_margin*1000:.1f} mV ({margins.margin_percentage:.1f}%)")
    print(f"  Critical limit: {margins.critical_limit}")
    print(f"  Status: {margins.margin_status}")

    if margins.warning:
        print("  ⚠ WARNING: Margin below 20% threshold")

else:
    print("  Status: ✗ FAIL")
    print(f"  Violations: {result.num_violations} ({result.violation_rate:.1%})")

    if result.upper_violations is not None:
        print(f"  Upper limit exceeded: {len(result.upper_violations)} samples")
        print(f"    Max overshoot: {result.max_value - spec.upper:.6f}V")

    if result.lower_violations is not None:
        print(f"  Lower limit violated: {len(result.lower_violations)} samples")
        print(f"    Max undershoot: {spec.lower - result.min_value:.6f}V")

# Visualization
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))

# Plot waveform
time = np.arange(len(power_rail.data)) / power_rail.metadata.sample_rate
ax.plot(time * 1e6, power_rail.data, label='Measured', linewidth=0.5)

# Plot limits
ax.axhline(y=spec.upper, color='r', linestyle='--', label='Upper limit', linewidth=2)
ax.axhline(y=spec.lower, color='r', linestyle='--', label='Lower limit', linewidth=2)

# Plot guardband
if spec.upper_guardband > 0:
    gb_upper = spec.upper - spec.upper_guardband
    gb_lower = spec.lower + spec.lower_guardband
    ax.axhline(y=gb_upper, color='orange', linestyle=':', label='Guardband', alpha=0.7)
    ax.axhline(y=gb_lower, color='orange', linestyle=':', alpha=0.7)
    ax.fill_between(time * 1e6, spec.lower, spec.upper, alpha=0.1, color='green')
    ax.fill_between(time * 1e6, gb_lower, gb_upper, alpha=0.1, color='yellow')

ax.set_xlabel('Time (μs)')
ax.set_ylabel('Voltage (V)')
ax.set_title(f'{spec.name} - Limit Test Results')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Example 4: Eye Diagram Mask Testing

```python
import oscura as osc
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import eye_mask, mask_test
from oscura.analyzers.eye import create_eye_diagram
import matplotlib.pyplot as plt
import numpy as np

# Load high-speed serial data
serial_data = osc.load("pcie_gen3_signal.wfm")

# Create eye diagram (overlaying bit periods)
bit_rate = 8e9  # 8 Gbps
eye_diagram = create_eye_diagram(
    serial_data,
    bit_rate=bit_rate,
    num_bits=1000
)

# Create standard eye mask (50% width, 40% height)
mask = eye_mask(
    eye_width=0.50,          # 50% of UI
    eye_height=0.40,         # 40% of amplitude
    center_height=0.30,
    unit_interval=1.0,
    amplitude=1.0
)

# Test eye diagram against mask
result = mask_test(
    eye_diagram,
    mask,
    normalize=True
)

# Print results
print(f"Eye Diagram Mask Test Results:")
print(f"  Status: {'✓ PASS' if result.passed else '✗ FAIL'}")
print(f"  Violations: {result.num_violations}")
print(f"  Violation rate: {result.violation_rate:.2%}")

if result.margin:
    print(f"  Margin: {result.margin:.4f}")

# Violations by region
print(f"\nViolations by region:")
for region, count in result.violations_by_region.items():
    print(f"  {region}: {count}")

# Visualize eye diagram with mask
fig, ax = plt.subplots(figsize=(10, 8))

# Plot eye diagram
ax.plot(eye_diagram.data, linewidth=0.1, alpha=0.3, color='blue')

# Overlay mask regions
for region in mask.regions:
    vertices = np.array(region.vertices + [region.vertices[0]])  # Close polygon

    if region.region_type == "violation":
        ax.fill(vertices[:, 0], vertices[:, 1],
                color='red', alpha=0.3, label=f'Mask: {region.name}')
        ax.plot(vertices[:, 0], vertices[:, 1],
                color='red', linewidth=2)
    else:
        ax.plot(vertices[:, 0], vertices[:, 1],
                color='green', linewidth=2, linestyle='--')

# Highlight violations
if result.violation_points:
    x_viol, y_viol = zip(*result.violation_points[:100])  # Limit for visibility
    ax.scatter(x_viol, y_viol, c='red', s=10, marker='x',
               label=f'Violations ({len(result.violation_points)})')

ax.set_xlabel('Time (UI)')
ax.set_ylabel('Normalized Amplitude')
ax.set_title(f'Eye Diagram with Mask - {"PASS" if result.passed else "FAIL"}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Pass/fail criteria
if not result.passed:
    print(f"\n✗ FAIL: Signal quality does not meet mask requirements")
    print(f"   Recommend: Check signal integrity, equalization, or termination")
else:
    print(f"\n✓ PASS: Signal meets mask requirements")
    if result.margin and result.margin < 0.1:
        print(f"   Note: Low margin detected, monitor signal quality")
```

## Best Practices

### Tolerance Selection

**Golden Reference Tolerance:**

- **Absolute tolerance**: Use when signal has fixed requirements (e.g., voltage levels)
- **Percentage tolerance**: Use when tolerance scales with signal amplitude
- **Sigma tolerance**: Use when averaging multiple samples with natural variation

```python
# Fixed voltage reference
golden = create_golden(trace, tolerance=0.050)  # 50mV absolute

# Scaling with amplitude
golden = create_golden(trace, tolerance_pct=5.0)  # 5% of range

# Statistical (from multiple samples)
golden = golden_from_average(samples, tolerance_sigma=3.0)  # 3σ
```

### Limit Testing Strategy

1. **Define specifications early** - Document limits before testing
2. **Use guardbands** - Add safety margin for manufacturing variation
3. **Perform margin analysis** - Monitor how close to limits you are
4. **Track trends** - Log margin over time to detect degradation

```python
# Good practice: Use guardband
spec = create_limit_spec(
    center=3.3,
    tolerance_pct=5,      # Specification limit
    guardband_pct=10,     # Additional manufacturing margin
    name="POWER_RAIL"
)

# Monitor margins
margins = margin_analysis(trace, spec, warning_threshold_pct=20)
if margins.margin_status == "warning":
    log_warning(f"Low margin on {spec.name}: {margins.margin_percentage:.1f}%")
```

### Mask Testing Tips

- **Normalize data** - Eye masks typically assume normalized amplitude
- **Choose appropriate mask** - Different standards have different masks
- **Sample adequately** - Ensure enough eye traces for statistical validity
- **Monitor margin** - Track how close waveform gets to mask boundary

### Performance Considerations

- **Array alignment** - Traces are automatically aligned to shorter length
- **Interpolation** - Use `interpolate=True` for golden testing with different lengths
- **Correlation alignment** - Auto-alignment compensates for trigger jitter
- **Batch processing** - Use batch functions for multiple comparisons

## Error Handling

```python
# Functions available at top-level: osc.function_name()
# Or import from oscura.utils.comparison import compare_traces, create_golden, check_limits
from oscura.core.exceptions import AnalysisError, LoaderError

try:
    # Load and compare
    trace1 = osc.load("file1.wfm")
    trace2 = osc.load("file2.wfm")
    result = compare_traces(trace1, trace2)

except LoaderError as e:
    print(f"Failed to load file: {e}")

except AnalysisError as e:
    print(f"Analysis failed: {e}")

except ValueError as e:
    # Invalid parameters (e.g., tolerance specification)
    print(f"Invalid parameters: {e}")

# Check for NaN/Inf in traces
if np.any(~np.isfinite(trace1.data)):
    print("Warning: Trace contains NaN or Inf values")
```

## See Also

- [Analysis API](analysis.md) - Waveform measurements
- [Visualization API](visualization.md) - Plotting comparison results
- [Reporting API](reporting.md) - Generating comparison reports
- [Loader API](loader.md) - Loading waveform files
- [Export API](export.md) - Exporting test results
