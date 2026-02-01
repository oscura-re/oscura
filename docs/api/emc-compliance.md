# EMC Compliance Testing API Reference

> **Version**: 0.9.0
> **Last Updated**: 2026-01-25

## Overview

Oscura provides comprehensive EMC (Electromagnetic Compatibility) and EMI (Electromagnetic Interference) compliance testing capabilities for regulatory standards including FCC, CE/CISPR, MIL-STD, and automotive EMC requirements. The compliance module enables limit mask testing, violation detection, and automated report generation for emissions and immunity testing.

## Quick Start

```python
import oscura as osc
from oscura.validation.compliance import (
    load_limit_mask, check_compliance,
    create_custom_mask, generate_compliance_report
)

# Load waveform and check against FCC limits
trace = osc.load("radiated_emissions.wfm")
mask = load_limit_mask("FCC_Part15_ClassB")
result = check_compliance(trace, mask)

# Generate compliance report
if result.passed:
    print(f"✓ PASS - Margin: {result.margin_to_limit:.1f} dB")
else:
    print(f"✗ FAIL - Violations: {result.violation_count}")

generate_compliance_report(result, "emc_report.html")
```

## Core Functions

### `check_compliance()`

Test signal against EMC limit mask for regulatory compliance.

```python
from oscura.validation.compliance import check_compliance, load_limit_mask

# Load limit mask
mask = load_limit_mask("FCC_Part15_ClassB")

# Check compliance
result = check_compliance(
    trace,                          # WaveformTrace or (freq, mag) tuple
    mask,                           # LimitMask to test against
    detector="peak",                # "peak", "quasi-peak", "average", "rms"
    frequency_range=None,           # Optional (f_min, f_max) Hz
    unit_conversion="V_to_dBuV"     # "V_to_dBuV", "W_to_dBm", or None
)

# Analyze results
print(f"Status: {result.status}")
print(f"Margin: {result.margin_to_limit:.1f} dB")
print(f"Worst frequency: {result.worst_frequency / 1e6:.3f} MHz")

if not result.passed:
    print(f"Violations: {result.violation_count}")
    for v in result.violations[:5]:
        print(f"  {v}")
```

**Parameters:**

| Parameter           | Type                                      | Default  | Description                              |
| ------------------- | ----------------------------------------- | -------- | ---------------------------------------- |
| `trace_or_spectrum` | `WaveformTrace \|tuple[NDArray, NDArray]` | required | Trace to analyze or (freq, mag) spectrum |
| `mask`              | `LimitMask`                               | required | EMC limit mask                           |
| `detector`          | `DetectorType \|str`                      | `"peak"` | Detector type for measurement            |
| `frequency_range`   | `tuple[float, float] \|None`              | `None`   | Frequency range to test (Hz)             |
| `unit_conversion`   | `str \|None`                              | `None`   | Convert units ("V_to_dBuV", "W_to_dBm")  |

**Returns:** `ComplianceResult`

**ComplianceResult Attributes:**

- `status: str` - "PASS" or "FAIL"
- `passed: bool` - True if test passed (property)
- `mask_name: str` - Name of limit mask used
- `violations: list[ComplianceViolation]` - List of violations
- `violation_count: int` - Number of violations (property)
- `margin_to_limit: float` - Minimum margin in dB (negative = failing)
- `worst_frequency: float` - Frequency with worst margin (Hz)
- `worst_margin: float` - Worst margin value in dB
- `spectrum_freq: NDArray[np.float64]` - Tested frequency array
- `spectrum_level: NDArray[np.float64]` - Measured level array
- `limit_level: NDArray[np.float64]` - Limit level array (interpolated)
- `detector: str` - Detector type used
- `metadata: dict[str, Any]` - Additional result metadata

**ComplianceViolation Attributes:**

- `frequency: float` - Violation frequency in Hz
- `measured_level: float` - Measured level in mask unit
- `limit_level: float` - Limit level at this frequency
- `excess_db: float` - Amount exceeding limit (positive = violation)
- `detector: str` - Detector type used
- `severity: str` - Severity classification

**DetectorType Values:**

- `PEAK` - Peak detector (highest instantaneous value)
- `QUASI_PEAK` - Quasi-peak detector (CISPR 16-1-1 weighted)
- `AVERAGE` - Average detector (Welch PSD-based)
- `RMS` - RMS detector

### `load_limit_mask()`

Load a built-in or custom EMC limit mask by name.

```python
from oscura.validation.compliance import load_limit_mask, AVAILABLE_MASKS

# Show available masks
print("Available masks:", AVAILABLE_MASKS)

# Load built-in mask
mask = load_limit_mask("FCC_Part15_ClassB")
print(f"Frequency range: {mask.frequency_range}")
print(f"Unit: {mask.unit}")
print(f"Distance: {mask.distance}m")

# Load custom mask from file
custom_mask = load_limit_mask("custom_limits.json")

# Load from custom directory
mask = load_limit_mask("MyMask", custom_path="./masks")
```

**Parameters:**

| Parameter     | Type                | Default  | Description                            |
| ------------- | ------------------- | -------- | -------------------------------------- |
| `name`        | `str`               | required | Mask name or path to JSON file         |
| `custom_path` | `str \|Path \|None` | `None`   | Directory containing custom mask files |

**Returns:** `LimitMask`

**Built-in Masks (AVAILABLE_MASKS):**

**FCC Standards:**

- `FCC_Part15_ClassA` - FCC Part 15 Class A (commercial) radiated emissions
- `FCC_Part15_ClassB` - FCC Part 15 Class B (residential) radiated emissions
- `FCC_Part15_ClassB_Conducted` - FCC Part 15 Class B conducted emissions

**CE/CISPR Standards:**

- `CE_CISPR22_ClassA` - CISPR 22 Class A radiated emissions
- `CE_CISPR22_ClassB` - CISPR 22 Class B radiated emissions
- `CE_CISPR32_ClassA` - CISPR 32 Class A radiated emissions
- `CE_CISPR32_ClassB` - CISPR 32 Class B radiated emissions
- `CE_CISPR32_ClassB_Conducted` - CISPR 32 Class B conducted emissions

**Military Standards:**

- `MIL_STD_461G_CE102` - MIL-STD-461G CE102 conducted emissions
- `MIL_STD_461G_RE102` - MIL-STD-461G RE102 radiated emissions
- `MIL_STD_461G_CS101` - MIL-STD-461G CS101 conducted susceptibility

**Mask Aliases:**

- `CISPR22`, `CISPR22_ClassB` → `CE_CISPR22_ClassB`
- `CISPR32`, `CISPR32_ClassB` → `CE_CISPR32_ClassB`

### `create_custom_mask()`

Create a custom EMC limit mask for specialized testing requirements.

```python
from oscura.validation.compliance import create_custom_mask

# Create automotive EMC mask (CISPR 25-style limits)
mask = create_custom_mask(
    name="CISPR_25_ClassB",
    frequencies=[150e3, 30e6, 108e6, 1000e6],  # Hz
    limits=[74, 54, 44, 44],                    # dBuV
    unit="dBuV",
    description="CISPR 25 Class B radiated emissions",
    detector="quasi-peak",
    distance=1.0,
    regulatory_body="ISO",
    document="CISPR 25:2016"
)

# Create custom ESD immunity test mask
esd_mask = create_custom_mask(
    name="IEC_61000_4_2_Level3",
    frequencies=[1e3, 1e6, 100e6],
    limits=[6, 6, 6],  # kV
    unit="kV",
    description="IEC 61000-4-2 ESD immunity Level 3",
    detector="peak"
)

# Mask with automatically sorted frequencies
mask = create_custom_mask(
    name="Custom",
    frequencies=[1000e6, 30e6, 100e6],  # Will be auto-sorted
    limits=[50, 60, 55]
)
```

**Parameters:**

| Parameter     | Type                            | Default  | Description                                |
| ------------- | ------------------------------- | -------- | ------------------------------------------ |
| `name`        | `str`                           | required | Mask name                                  |
| `frequencies` | `list[float] \|NDArray`         | required | Frequency points in Hz                     |
| `limits`      | `list[float] \|NDArray`         | required | Limit values in specified unit             |
| `unit`        | `str`                           | `"dBuV"` | Limit unit ("dBuV", "dBm", "dBuV/m", etc.) |
| `description` | `str`                           | `""`     | Human-readable description                 |
| `**kwargs`    | Additional LimitMask attributes |          | `detector`, `distance`, etc.               |

**Returns:** `LimitMask`

**Raises:** `ValueError` if frequencies and limits have different lengths

**Note:** Frequencies are automatically sorted if not provided in ascending order.

### `generate_compliance_report()`

Generate formatted EMC compliance test report in HTML, PDF, Markdown, or JSON.

```python
from oscura.validation.compliance import generate_compliance_report, ComplianceReportFormat

# Generate HTML report with plot
report_path = generate_compliance_report(
    result,                         # ComplianceResult from check_compliance()
    "emc_report.html",
    format="html",                  # "html", "pdf", "markdown", "json"
    include_plot=True,              # Include spectrum vs limit plot
    title="Product EMC Test Report",
    company_name="Acme Electronics",
    dut_info={
        "Model": "XYZ-100",
        "Serial": "12345",
        "Test Date": "2026-01-08",
        "Lab": "EMC Test Lab A"
    }
)

# Generate PDF report
generate_compliance_report(
    result,
    "emc_report.pdf",
    format=ComplianceReportFormat.PDF,
    include_plot=True,
    title="FCC Part 15 Compliance Test"
)

# Generate JSON for automated processing
generate_compliance_report(
    result,
    "emc_report.json",
    format="json"
)
```

**Parameters:**

| Parameter      | Type                           | Default                       | Description                            |
| -------------- | ------------------------------ | ----------------------------- | -------------------------------------- |
| `result`       | `ComplianceResult`             | required                      | Compliance test result                 |
| `output_path`  | `str \|Path`                   | required                      | Output file path                       |
| `format`       | `ComplianceReportFormat \|str` | `ComplianceReportFormat.HTML` | Report format                          |
| `include_plot` | `bool`                         | `True`                        | Include spectrum/limit plot (HTML/PDF) |
| `title`        | `str \|None`                   | `"EMC Compliance Report"`     | Report title                           |
| `company_name` | `str \|None`                   | `None`                        | Company name for header                |
| `dut_info`     | `dict[str, str] \|None`        | `None`                        | Device Under Test information          |

**Returns:** `Path` - Path to generated report

**Report Formats:**

- `HTML` - Interactive HTML with embedded SVG plot
- `PDF` - PDF report (requires weasyprint)
- `MARKDOWN` - Markdown text report
- `JSON` - Machine-readable JSON data

## LimitMask Class

EMC limit mask definition with interpolation capabilities.

**Attributes:**

- `name: str` - Standard name (e.g., "FCC_Part15_ClassB")
- `description: str` - Human-readable description
- `frequency: NDArray[np.float64]` - Frequency points in Hz
- `limit: NDArray[np.float64]` - Limit values in specified unit
- `unit: str` - Limit unit ("dBuV", "dBm", "dBuV/m", etc.)
- `standard: str` - Standard designation
- `distance: float` - Measurement distance in meters
- `detector: str` - Required detector type
- `regulatory_body: str` - Regulatory body (FCC, CE, MIL, ISO)
- `document: str` - Reference document
- `metadata: dict[str, Any]` - Additional metadata

**Properties:**

- `frequency_range: tuple[float, float]` - (min, max) frequency range

**Methods:**

```python
# Get limit at specific frequency
limit_at_100mhz = mask.get_limit_at_frequency(100e6)

# Interpolate to frequency array
freq_array = np.linspace(30e6, 1000e6, 1000)
limit_array = mask.interpolate(freq_array)

# Serialize to dictionary
mask_dict = mask.to_dict()

# Load from dictionary
mask = LimitMask.from_dict(mask_dict)
```

## Practical Examples

### Example 1: Testing Against CISPR 25 Automotive EMC Limits

```python
import oscura as osc
from oscura.validation.compliance import create_custom_mask, check_compliance, generate_compliance_report
import numpy as np

# Create CISPR 25 Class B radiated emissions mask (automotive)
cispr25_mask = create_custom_mask(
    name="CISPR_25_ClassB_Radiated",
    frequencies=[
        150e3, 5.9e6, 6.2e6, 30e6, 54e6,
        88e6, 108e6, 174e6, 230e6, 1000e6
    ],
    limits=[
        74,    # 150 kHz - 5.9 MHz: 74 dBuV
        74,    # Peak at 5.9 MHz
        60,    # 6.2 MHz: 60 dBuV
        54,    # 30 MHz: 54 dBuV
        50,    # 54 MHz: 50 dBuV
        48,    # 88 MHz: 48 dBuV
        44,    # 108 MHz - 230 MHz: 44 dBuV
        44,
        44,
        44     # Up to 1 GHz
    ],
    unit="dBuV",
    description="CISPR 25 Class B Radiated Emissions (Broadband)",
    detector="quasi-peak",
    distance=1.0,  # 1 meter for automotive
    regulatory_body="ISO",
    document="CISPR 25:2016 Section 6.5"
)

# Load automotive component emissions data
trace = osc.load("automotive_ecu_emissions.wfm")

# Check compliance
result = check_compliance(
    trace,
    cispr25_mask,
    detector="quasi-peak",
    frequency_range=(150e3, 1000e6)  # Test full automotive range
)

# Analyze results
print(f"CISPR 25 Compliance Test")
print(f"Status: {result.status}")
print(f"Margin to limit: {result.margin_to_limit:.1f} dB")

if not result.passed:
    print(f"\nViolations detected: {result.violation_count}")
    print("\nWorst violations:")
    for v in sorted(result.violations, key=lambda x: x.excess_db, reverse=True)[:5]:
        print(f"  {v.frequency / 1e6:.3f} MHz: {v.excess_db:.1f} dB over limit")
else:
    print(f"✓ PASS - Minimum margin: {result.margin_to_limit:.1f} dB")
    print(f"  at {result.worst_frequency / 1e6:.3f} MHz")

# Generate detailed report
generate_compliance_report(
    result,
    "cispr25_compliance_report.html",
    title="CISPR 25 Automotive EMC Compliance Test",
    company_name="Automotive Components Inc.",
    dut_info={
        "Device": "Engine Control Unit (ECU)",
        "Part Number": "ECU-2000",
        "Test Standard": "CISPR 25:2016",
        "Test Date": "2026-01-08",
        "Test Engineer": "J. Smith",
        "Temperature": "25°C",
        "Supply Voltage": "12V"
    }
)
```

### Example 2: FCC Part 15 Radiated Emissions Testing

```python
import oscura as osc
from oscura.validation.compliance import load_limit_mask, check_compliance
import matplotlib.pyplot as plt

# Load radiated emissions measurement
trace = osc.load("radiated_emissions_3m.wfm")

# Load FCC Part 15 Class B limit (residential equipment)
fcc_mask = load_limit_mask("FCC_Part15_ClassB")

print(f"FCC Part 15 Class B Radiated Emissions")
print(f"Frequency range: {fcc_mask.frequency_range[0] / 1e6:.1f} - "
      f"{fcc_mask.frequency_range[1] / 1e6:.1f} MHz")
print(f"Measurement distance: {fcc_mask.distance}m")
print(f"Detector: {fcc_mask.detector}")

# Perform compliance test
result = check_compliance(
    trace,
    fcc_mask,
    detector="quasi-peak",
    unit_conversion="V_to_dBuV"
)

# Display results
print(f"\nTest Result: {result.status}")
print(f"Margin to limit: {result.margin_to_limit:.1f} dB")
print(f"Worst case: {result.worst_frequency / 1e6:.1f} MHz "
      f"({result.worst_margin:.1f} dB margin)")

if result.passed:
    print("\n✓ Device complies with FCC Part 15 Class B limits")
else:
    print(f"\n✗ FAIL: {result.violation_count} violations detected")
    print("\nViolation summary:")
    for v in result.violations:
        freq_mhz = v.frequency / 1e6
        print(f"  {freq_mhz:7.3f} MHz: {v.measured_level:5.1f} dBµV/m "
              f"(limit: {v.limit_level:5.1f} dBµV/m, "
              f"excess: +{v.excess_db:.1f} dB)")

# Plot spectrum vs limit
plt.figure(figsize=(12, 6))
plt.semilogx(result.spectrum_freq / 1e6, result.spectrum_level,
             'b-', linewidth=1.5, label='Measured Spectrum')
plt.semilogx(result.spectrum_freq / 1e6, result.limit_level,
             'r--', linewidth=2, label='FCC Limit')

# Highlight violations
if result.violations:
    viol_freq = [v.frequency / 1e6 for v in result.violations]
    viol_level = [v.measured_level for v in result.violations]
    plt.scatter(viol_freq, viol_level, c='red', s=50,
                marker='x', zorder=5, label='Violations')

plt.xlabel('Frequency (MHz)')
plt.ylabel('Field Strength (dBµV/m)')
plt.title('FCC Part 15 Class B Radiated Emissions @ 3m')
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('fcc_part15_spectrum.png', dpi=150)
plt.show()

# Generate official compliance report
generate_compliance_report(
    result,
    "fcc_part15_compliance.html",
    title="FCC Part 15 Class B Compliance Test",
    dut_info={
        "Product": "Wireless Router Model WR-5000",
        "FCC ID": "ABC123456",
        "Test Standard": "FCC Part 15 Subpart B",
        "Test Method": "ANSI C63.4-2014",
        "Test Distance": "3 meters",
        "Antenna": "Biconical + Log Periodic",
        "Test Lab": "EMC Lab XYZ (FCC Accredited)"
    }
)
```

### Example 3: IEC 61000-4-2 ESD Immunity Testing

```python
import oscura as osc
from oscura.validation.compliance import create_custom_mask, check_compliance
import numpy as np

# Create IEC 61000-4-2 ESD immunity mask (time-domain waveform envelope)
# ESD pulse characteristics per IEC 61000-4-2
esd_mask = create_custom_mask(
    name="IEC_61000_4_2_Level3",
    frequencies=[
        0,      # Start
        1e-9,   # 1ns rise time
        30e-9,  # First peak
        60e-9,  # Decay
        100e-9  # End of interest
    ],
    limits=[
        0,      # 0V start
        6000,   # 6kV peak (Level 3)
        4000,   # Decay envelope
        1000,   # Continued decay
        0       # Return to baseline
    ],
    unit="V",
    description="IEC 61000-4-2 ESD Immunity Test Level 3 (Contact Discharge)",
    detector="peak",
    distance=0,
    regulatory_body="IEC",
    document="IEC 61000-4-2:2008"
)

# Load ESD pulse measurement
esd_pulse = osc.load("esd_discharge_contact.wfm")

# For time-domain testing, create (time, voltage) tuple
time = np.arange(len(esd_pulse.data)) / esd_pulse.metadata.sample_rate
voltage = np.abs(esd_pulse.data)  # Use absolute value for envelope

# Check if pulse stays within envelope
result = check_compliance(
    (time, voltage),
    esd_mask,
    detector="peak"
)

# Analyze immunity test results
print(f"IEC 61000-4-2 ESD Immunity Test - Level 3 (±6kV)")
print(f"Status: {result.status}")

if result.passed:
    print(f"✓ PASS: Device immune to {6} kV ESD")
    print(f"  Margin: {result.margin_to_limit:.1f} dB")
else:
    print(f"✗ FAIL: Device susceptible to ESD")
    print(f"  Peak voltage exceeded limit by: "
          f"{-result.worst_margin:.1f} dB")

# Alternative: Test ESD spectrum for frequency content analysis
# This checks if ESD generator meets spectral requirements
esd_spectrum_mask = create_custom_mask(
    name="IEC_61000_4_2_Spectrum",
    frequencies=[10e6, 100e6, 500e6, 1000e6],
    limits=[40, 35, 30, 25],  # Approximate spectral content limits
    unit="dBV",
    description="IEC 61000-4-2 ESD Generator Spectrum Requirements"
)

# Compute spectrum and verify ESD generator
from oscura.analyzers.spectral import fft

freq, mag = fft(esd_pulse)
spectrum_result = check_compliance(
    (freq, np.abs(mag)),
    esd_spectrum_mask,
    detector="peak",
    frequency_range=(10e6, 1000e6)
)

print(f"\nESD Generator Verification: {spectrum_result.status}")
if spectrum_result.passed:
    print("✓ ESD generator meets spectral requirements")
```

### Example 4: Creating Custom Compliance Masks

```python
from oscura.validation.compliance import create_custom_mask, LimitMask
import numpy as np
import json

# Example 1: Multi-segment piecewise linear mask
# Typical for complex EMC requirements with frequency-dependent limits
mask = create_custom_mask(
    name="Custom_MultiSegment",
    frequencies=[
        10e3, 150e3,  # Below AM band: strict
        150e3, 30e6,  # AM band: very strict
        30e6, 88e6,   # Below FM: moderate
        88e6, 108e6,  # FM band: strict
        108e6, 1e9    # Above FM: relaxed
    ],
    limits=[
        85, 70,       # 10-150 kHz
        70, 60,       # 150 kHz - 30 MHz
        60, 55,       # 30-88 MHz
        55, 50,       # 88-108 MHz (FM)
        50, 45        # 108 MHz - 1 GHz
    ],
    unit="dBuV",
    description="Custom Multi-Band EMC Limit",
    detector="quasi-peak",
    regulatory_body="Internal",
    document="Internal Standard v2.0"
)

# Example 2: Logarithmic frequency sweep mask
# Common for wide bandwidth testing
frequencies = np.logspace(4, 9, 50)  # 10 kHz to 1 GHz, 50 points
limits = 80 - 10 * np.log10(frequencies / 1e6)  # Decreasing with frequency

log_mask = create_custom_mask(
    name="Logarithmic_Sweep",
    frequencies=frequencies,
    limits=np.clip(limits, 30, 80),  # Clip to reasonable range
    unit="dBuV/m",
    description="Logarithmic frequency sweep limit"
)

# Example 3: Notch limits (protect specific frequencies)
# Useful for protecting sensitive frequencies (GPS, WiFi, etc.)
base_limit = 50  # Base limit in dBuV

# Create frequency points with higher limits except at notches
freq_points = []
limit_points = []

# Main band: 1 MHz to 2 GHz
for f in np.logspace(6, 9, 100):
    freq_points.append(f)

    # GPS L1 notch at 1575.42 MHz (±10 MHz)
    if 1565e6 <= f <= 1585e6:
        limit_points.append(30)  # Very strict at GPS
    # WiFi 2.4 GHz notch (±50 MHz)
    elif 2400e6 <= f <= 2500e6:
        limit_points.append(35)  # Strict at WiFi
    else:
        limit_points.append(base_limit)  # Normal limit

notch_mask = create_custom_mask(
    name="Protected_Bands",
    frequencies=freq_points,
    limits=limit_points,
    unit="dBuV/m",
    description="EMC limits with protected frequency bands (GPS, WiFi)",
    detector="average"
)

# Example 4: Save custom mask to JSON for reuse
custom_mask_data = mask.to_dict()
with open("custom_emc_mask.json", "w") as f:
    json.dump(custom_mask_data, f, indent=2)

# Load custom mask from JSON
with open("custom_emc_mask.json") as f:
    loaded_data = json.load(f)
loaded_mask = LimitMask.from_dict(loaded_data)

print("Custom mask created and saved successfully")
print(f"Frequency range: {loaded_mask.frequency_range[0] / 1e6:.3f} - "
      f"{loaded_mask.frequency_range[1] / 1e6:.3f} MHz")
```

### Example 5: Generating Pass/Fail Reports

```python
import oscura as osc
from oscura.validation.compliance import load_limit_mask, check_compliance, generate_compliance_report
from pathlib import Path

# Test multiple devices and generate reports
devices = [
    ("Unit_001", "device_001_emissions.wfm"),
    ("Unit_002", "device_002_emissions.wfm"),
    ("Unit_003", "device_003_emissions.wfm"),
    ("Unit_004", "device_004_emissions.wfm"),
    ("Unit_005", "device_005_emissions.wfm"),
]

# Load compliance mask
mask = load_limit_mask("FCC_Part15_ClassB")

# Create reports directory
reports_dir = Path("emc_reports")
reports_dir.mkdir(exist_ok=True)

# Test each device
results_summary = []

for unit_id, filename in devices:
    print(f"\nTesting {unit_id}...")

    # Load trace
    trace = osc.load(filename)

    # Check compliance
    result = check_compliance(
        trace,
        mask,
        detector="quasi-peak",
        unit_conversion="V_to_dBuV"
    )

    # Store summary
    results_summary.append({
        "unit_id": unit_id,
        "status": result.status,
        "margin_db": result.margin_to_limit,
        "violations": result.violation_count,
        "worst_freq_mhz": result.worst_frequency / 1e6
    })

    # Generate individual HTML report
    generate_compliance_report(
        result,
        reports_dir / f"{unit_id}_emc_report.html",
        title=f"FCC Part 15 Compliance Test - {unit_id}",
        dut_info={
            "Unit ID": unit_id,
            "Test Standard": "FCC Part 15 Class B",
            "Test Date": "2026-01-08"
        }
    )

    # Generate JSON for data processing
    generate_compliance_report(
        result,
        reports_dir / f"{unit_id}_emc_data.json",
        format="json"
    )

    # Print result
    status_symbol = "✓" if result.passed else "✗"
    print(f"  {status_symbol} {result.status} - Margin: {result.margin_to_limit:.1f} dB")

# Generate summary report
print("\n" + "="*60)
print("EMC COMPLIANCE TEST SUMMARY")
print("="*60)
print(f"{'Unit ID':<12} {'Status':<8} {'Margin (dB)':<12} {'Violations':<12} {'Worst Freq (MHz)'}")
print("-"*60)

pass_count = 0
for summary in results_summary:
    status_symbol = "✓" if summary["status"] == "PASS" else "✗"
    print(f"{summary['unit_id']:<12} {status_symbol} {summary['status']:<6} "
          f"{summary['margin_db']:>10.1f}  {summary['violations']:>10}  "
          f"{summary['worst_freq_mhz']:>16.3f}")
    if summary["status"] == "PASS":
        pass_count += 1

print("-"*60)
print(f"Pass Rate: {pass_count}/{len(devices)} ({pass_count/len(devices)*100:.1f}%)")
print(f"\nDetailed reports saved to: {reports_dir.absolute()}")
```

## Advanced Usage

### Using Pre-Computed Spectrum

For efficiency when testing multiple masks:

```python
import oscura as osc
from oscura.analyzers.spectral import fft
from oscura.validation.compliance import load_limit_mask, check_compliance
import numpy as np

# Load trace once
trace = osc.load("emissions.wfm")

# Compute spectrum once
freq, mag = fft(trace)
spectrum = (freq, np.abs(mag))

# Test against multiple standards
masks = [
    "FCC_Part15_ClassB",
    "CE_CISPR32_ClassB",
    "MIL_STD_461G_RE102"
]

for mask_name in masks:
    mask = load_limit_mask(mask_name)
    result = check_compliance(
        spectrum,  # Reuse spectrum
        mask,
        detector="peak",
        unit_conversion="V_to_dBuV"
    )
    print(f"{mask_name}: {result.status} (margin: {result.margin_to_limit:.1f} dB)")
```

### Custom Detector Implementation

For specialized detector types:

```python
from oscura.validation.compliance import check_compliance, load_limit_mask
import numpy as np

# Implement custom detector processing
def custom_detector(trace):
    """Custom weighted averaging detector."""
    from oscura.analyzers.spectral import fft

    freq, mag = fft(trace)

    # Apply custom weighting (e.g., for specific standard)
    weights = np.exp(-freq / 100e6)  # Example: frequency-dependent weighting
    weighted_mag = np.abs(mag) * weights

    return freq, weighted_mag

# Use custom detector
trace = osc.load("emissions.wfm")
freq, mag = custom_detector(trace)

mask = load_limit_mask("FCC_Part15_ClassB")
result = check_compliance(
    (freq, mag),
    mask,
    detector="peak"  # Detector type for metadata
)
```

### Margin Analysis and Trending

Track compliance margins over time:

```python
import oscura as osc
from oscura.validation.compliance import load_limit_mask, check_compliance
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load historical test data
test_dates = [datetime.now() - timedelta(days=d) for d in range(30, 0, -5)]
test_files = [f"emissions_day{30-i}.wfm" for i in range(0, 30, 5)]

mask = load_limit_mask("FCC_Part15_ClassB")
margins = []
worst_freqs = []

for date, filename in zip(test_dates, test_files):
    trace = osc.load(filename)
    result = check_compliance(trace, mask)
    margins.append(result.margin_to_limit)
    worst_freqs.append(result.worst_frequency / 1e6)

# Plot margin trend
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Margin over time
ax1.plot(test_dates, margins, 'o-', linewidth=2, markersize=8)
ax1.axhline(y=0, color='r', linestyle='--', label='Compliance Threshold')
ax1.axhline(y=6, color='orange', linestyle=':', label='Warning Level (6 dB)')
ax1.set_ylabel('Margin to Limit (dB)')
ax1.set_title('EMC Compliance Margin Trend')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Worst frequency over time
ax2.plot(test_dates, worst_freqs, 's-', linewidth=2, markersize=8, color='red')
ax2.set_xlabel('Date')
ax2.set_ylabel('Worst Case Frequency (MHz)')
ax2.set_title('Worst Case Frequency Trend')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('emc_margin_trend.png', dpi=150)
plt.show()

# Alert if margin is degrading
if margins[-1] < margins[0] - 3:
    print("⚠ WARNING: EMC margin has degraded by >3 dB")
    print(f"  Initial: {margins[0]:.1f} dB")
    print(f"  Current: {margins[-1]:.1f} dB")
```

## Best Practices

### Detector Selection

**Peak Detector:**

- Use for initial screening and worst-case analysis
- Required by some military standards (MIL-STD-461)
- Fastest measurement, conservative results

**Quasi-Peak Detector:**

- Required for most CE/CISPR standards
- Accounts for repetition rate (per CISPR 16-1-1)
- More representative of human perception/interference

**Average Detector:**

- Required for some conducted emissions tests
- Better correlation with long-term interference
- Less sensitive to transients

**RMS Detector:**

- Used for power measurements
- Good for broadband noise analysis

```python
# Example: Compare detector types
from oscura.validation.compliance import load_limit_mask, check_compliance

trace = osc.load("emissions.wfm")
mask = load_limit_mask("FCC_Part15_ClassB")

for detector in ["peak", "quasi-peak", "average", "rms"]:
    result = check_compliance(trace, mask, detector=detector)
    print(f"{detector:12s}: {result.status} (margin: {result.margin_to_limit:+.1f} dB)")
```

### Frequency Range Selection

Test only relevant frequency ranges to improve performance:

```python
# Automotive: 150 kHz - 2.5 GHz
result = check_compliance(trace, mask, frequency_range=(150e3, 2.5e9))

# Consumer electronics: 9 kHz - 1 GHz
result = check_compliance(trace, mask, frequency_range=(9e3, 1e9))

# Military: Often 10 kHz - 18 GHz
result = check_compliance(trace, mask, frequency_range=(10e3, 18e9))
```

### Test Documentation

Always document test conditions:

```python
dut_info = {
    "Product": "Device Model",
    "Serial Number": "SN12345",
    "Firmware Version": "v2.3.1",
    "Test Standard": "FCC Part 15B / ANSI C63.4",
    "Test Date": "2026-01-08",
    "Test Engineer": "Engineer Name",
    "Test Lab": "Lab Name (Accreditation #)",
    "Test Equipment": "Spectrum Analyzer, Antennas, etc.",
    "Environmental Conditions": "23°C, 45% RH",
    "Supply Voltage": "Nominal ±10%",
    "Operating Mode": "Normal operation with load",
    "Configuration": "All peripherals connected"
}

generate_compliance_report(result, "report.html", dut_info=dut_info)
```

## Error Handling

```python
from oscura.validation.compliance import load_limit_mask, check_compliance
from oscura.core.exceptions import AnalysisError

try:
    # Load mask
    mask = load_limit_mask("FCC_Part15_ClassB")

    # Check compliance
    result = check_compliance(trace, mask)

except ValueError as e:
    print(f"Invalid parameter: {e}")

except FileNotFoundError as e:
    print(f"Mask file not found: {e}")

except AnalysisError as e:
    print(f"Analysis failed: {e}")

# Validate spectrum data
if len(result.spectrum_freq) == 0:
    print("Warning: No data in mask frequency range")

# Check for NaN/Inf
import numpy as np
if np.any(~np.isfinite(result.spectrum_level)):
    print("Warning: Invalid values in spectrum")
```

## See Also

- [Analysis API](analysis.md) - Spectral analysis and FFT
- [Comparison & Limits API](comparison-and-limits.md) - Limit testing and masks
- [Reporting API](reporting.md) - Report generation
- [Visualization API](visualization.md) - Plotting spectra

## References

### EMC Standards

- **FCC Part 15** - Code of Federal Regulations Title 47 Part 15
- **CISPR 16-1-1** - Specification for radio disturbance and immunity measuring apparatus
- **CISPR 22** - Information technology equipment - Radio disturbance characteristics
- **CISPR 25** - Vehicles, boats and internal combustion engines - Radio disturbance characteristics
- **CISPR 32** - Electromagnetic compatibility of multimedia equipment
- **IEC 61000-4-2** - Electrostatic discharge immunity test
- **IEC 61000-4-7** - General guide on harmonics and interharmonics measurements
- **MIL-STD-461G** - Requirements for the control of electromagnetic interference characteristics
- **ANSI C63.4** - American National Standard for Methods of Measurement of Radio-Noise Emissions
- **ANSI C63.2** - American National Standard for Electromagnetic Noise and Field Strength Instrumentation
