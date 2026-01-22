# Component Analysis API Reference

> **Version**: 0.4.0 | **Last Updated**: 2026-01-19

## Overview

The Oscura Component Analysis API provides comprehensive tools for extracting electrical parameters from waveform measurements. This includes Time Domain Reflectometry (TDR) impedance profiling, reactive component characterization (capacitance and inductance), parasitic extraction, and transmission line analysis.

Component analysis is essential for:

- **PCB Trace Characterization** - Measure impedance, propagation delay, and discontinuities
- **Transmission Line Analysis** - Extract characteristic impedance and velocity factors
- **Component Measurement** - Determine capacitance and inductance from I/V waveforms
- **Parasitic Extraction** - Model parasitic R, L, C parameters
- **Quality Control** - Verify manufactured traces meet impedance specifications
- **Signal Integrity** - Identify impedance discontinuities and mismatches

All functions are accessible from the top-level API:

```python
import oscura as osc

# TDR impedance analysis
z0, profile = osc.extract_impedance(tdr_trace)

# Component measurements
C = osc.measure_capacitance(voltage_trace, current_trace)
L = osc.measure_inductance(voltage_trace, current_trace)

# Transmission line parameters
z0 = osc.characteristic_impedance(tdr_trace)
delay = osc.propagation_delay(tdr_trace)
vf = osc.velocity_factor(tdr_trace, line_length=0.1)
```

## Quick Start

### TDR Impedance Profiling

```python
import oscura as osc

# Load TDR measurement
tdr_trace = osc.load("tdr_measurement.wfm")

# Extract impedance profile
z0, profile = osc.extract_impedance(tdr_trace, z0_source=50.0)

print(f"Characteristic impedance: {z0:.1f} ohms")
print(f"Mean impedance: {profile.mean_impedance:.1f} ohms")
print(f"Impedance range: {profile.min_impedance:.1f} - {profile.max_impedance:.1f} ohms")

# Plot impedance vs distance
import matplotlib.pyplot as plt
plt.plot(profile.distance * 1000, profile.impedance)
plt.xlabel('Distance (mm)')
plt.ylabel('Impedance (ohms)')
plt.title('TDR Impedance Profile')
plt.grid(True)
plt.show()
```

### Capacitance Measurement

```python
import oscura as osc

# Load voltage and current waveforms
voltage = osc.load("cap_voltage.wfm")
current = osc.load("cap_current.wfm")

# Measure capacitance
C_result = osc.measure_capacitance(voltage, current, method="charge")

print(f"Capacitance: {C_result.capacitance * 1e12:.2f} pF")
print(f"ESR: {C_result.esr:.2f} ohms")
print(f"Confidence: {C_result.confidence:.1%}")
```

### Transmission Line Characterization

```python
import oscura as osc

# Load TDR trace
tdr_trace = osc.load("pcb_trace.wfm")

# Get transmission line parameters
z0 = osc.characteristic_impedance(tdr_trace)
delay = osc.propagation_delay(tdr_trace)
vf = osc.velocity_factor(tdr_trace, line_length=0.050)  # 50mm trace

print(f"Z0: {z0:.1f} ohms")
print(f"Propagation delay: {delay * 1e9:.2f} ns")
print(f"Velocity factor: {vf:.3f}")
```

## TDR Impedance Analysis

### osc.extract_impedance()

Extract impedance profile from Time Domain Reflectometry (TDR) waveform.

**Signature:**

```python
osc.extract_impedance(
    trace,
    *,
    z0_source=50.0,
    velocity=None,
    velocity_factor=0.66,
    start_time=None,
    end_time=None
) -> tuple[float, ImpedanceProfile]
```

**Parameters:**

|Parameter|Type|Default|Description|
|---|---|---|---|
|`trace`|WaveformTrace|required|TDR reflection waveform|
|`z0_source`|float|`50.0`|Source/reference impedance (ohms)|
|`velocity`|float \|None|`None`|Propagation velocity (m/s), auto-calculated if None|
|`velocity_factor`|float|`0.66`|Fraction of speed of light (default for FR4)|
|`start_time`|float \|None|`None`|Start time for analysis window (seconds)|
|`end_time`|float \|None|`None`|End time for analysis window (seconds)|

**Returns:**

- `tuple[float, ImpedanceProfile]`: Tuple of (characteristic_impedance, impedance_profile)

**ImpedanceProfile Attributes:**

|Attribute|Type|Description|
|---|---|---|
|`distance`|ndarray|Distance axis in meters|
|`time`|ndarray|Time axis in seconds|
|`impedance`|ndarray|Impedance values in ohms|
|`z0_source`|float|Source impedance used (ohms)|
|`velocity`|float|Propagation velocity used (m/s)|
|`statistics`|dict|Additional statistics|
|`mean_impedance`|float|Mean impedance value (property)|
|`max_impedance`|float|Maximum impedance value (property)|
|`min_impedance`|float|Minimum impedance value (property)|

**Example:**

```python
import oscura as osc

# Load TDR measurement
tdr_trace = osc.load("tdr_50ohm_line.wfm")

# Extract impedance (default FR4 parameters)
z0, profile = osc.extract_impedance(tdr_trace, z0_source=50.0)

print(f"Characteristic impedance: {z0:.2f} ohms")
print(f"Mean impedance: {profile.mean_impedance:.2f} ohms")
print(f"Std deviation: {profile.statistics['z0_std']:.2f} ohms")

# For Rogers 4003C substrate (Er=3.38)
import numpy as np
vf_rogers = 1 / np.sqrt(3.38)
z0, profile = osc.extract_impedance(
    tdr_trace,
    z0_source=50.0,
    velocity_factor=vf_rogers
)

# Analyze specific region
z0, profile = osc.extract_impedance(
    tdr_trace,
    start_time=1e-9,   # 1 ns
    end_time=5e-9      # 5 ns
)
```

**References:**

- IPC-TM-650 2.5.5.7: Characteristic Impedance of Lines on PCBs
- IEEE 370-2020: Electrical Characterization of Interconnects

---

### osc.impedance_profile()

Get impedance profile from TDR waveform. Convenience function that returns only the impedance profile.

**Signature:**

```python
osc.impedance_profile(
    trace,
    *,
    z0_source=50.0,
    velocity_factor=0.66,
    smooth_window=0
) -> ImpedanceProfile
```

**Parameters:**

|Parameter|Type|Default|Description|
|---|---|---|---|
|`trace`|WaveformTrace|required|TDR reflection waveform|
|`z0_source`|float|`50.0`|Source/reference impedance (ohms)|
|`velocity_factor`|float|`0.66`|Fraction of speed of light|
|`smooth_window`|int|`0`|Smoothing window size (0 = no smoothing)|

**Returns:**

- `ImpedanceProfile`: Impedance profile object

**Example:**

```python
import oscura as osc
import matplotlib.pyplot as plt

# Get smoothed impedance profile
profile = osc.impedance_profile(
    tdr_trace,
    z0_source=50.0,
    smooth_window=5  # 5-point moving average
)

# Plot impedance vs distance
plt.figure(figsize=(10, 6))
plt.plot(profile.distance * 1000, profile.impedance)
plt.axhline(y=50, color='r', linestyle='--', label='Target Z0')
plt.xlabel('Distance (mm)')
plt.ylabel('Impedance (ohms)')
plt.title('TDR Impedance Profile')
plt.legend()
plt.grid(True)
plt.show()
```

---

### osc.discontinuity_analysis()

Analyze impedance discontinuities in TDR waveform. Detects and characterizes impedance changes along a transmission line.

**Signature:**

```python
osc.discontinuity_analysis(
    trace,
    *,
    z0_source=50.0,
    velocity_factor=0.66,
    threshold=5.0,
    min_separation=1e-12
) -> list[Discontinuity]
```

**Parameters:**

|Parameter|Type|Default|Description|
|---|---|---|---|
|`trace`|WaveformTrace|required|TDR reflection waveform|
|`z0_source`|float|`50.0`|Source/reference impedance (ohms)|
|`velocity_factor`|float|`0.66`|Fraction of speed of light|
|`threshold`|float|`5.0`|Minimum impedance change to detect (ohms)|
|`min_separation`|float|`1e-12`|Minimum time between discontinuities (sec)|

**Returns:**

- `list[Discontinuity]`: List of detected discontinuities

**Discontinuity Attributes:**

|Attribute|Type|Description|
|---|---|---|
|`position`|float|Position in meters|
|`time`|float|Time position in seconds|
|`impedance_before`|float|Impedance before discontinuity (ohms)|
|`impedance_after`|float|Impedance after discontinuity (ohms)|
|`magnitude`|float|Magnitude of change (ohms)|
|`reflection_coeff`|float|Reflection coefficient (rho)|
|`discontinuity_type`|str|Type: "capacitive", "inductive", "resistive", "unknown"|

**Example:**

```python
import oscura as osc

# Detect discontinuities
discontinuities = osc.discontinuity_analysis(
    tdr_trace,
    threshold=3.0,  # Detect changes > 3 ohms
    min_separation=100e-12  # Minimum 100 ps apart
)

print(f"Found {len(discontinuities)} discontinuities:\n")

for i, disc in enumerate(discontinuities, 1):
    print(f"Discontinuity {i}:")
    print(f"  Position: {disc.position * 1000:.2f} mm")
    print(f"  Time: {disc.time * 1e9:.2f} ns")
    print(f"  Impedance change: {disc.impedance_before:.1f} → {disc.impedance_after:.1f} ohms")
    print(f"  Magnitude: {disc.magnitude:+.1f} ohms")
    print(f"  Type: {disc.discontinuity_type}")
    print(f"  Reflection coeff: {disc.reflection_coeff:.4f}\n")
```

**Use Cases:**

- **Via Detection** - Identify vias causing impedance changes
- **Connector Quality** - Detect impedance mismatches at connectors
- **Manufacturing Defects** - Find width variations or dielectric issues
- **Stub Analysis** - Locate and characterize transmission line stubs

---

## Reactive Component Measurement

### osc.measure_capacitance()

Measure capacitance from voltage/current waveforms using various methods.

**Signature:**

```python
osc.measure_capacitance(
    voltage_trace,
    current_trace=None,
    *,
    method="charge",
    resistance=None
) -> CapacitanceMeasurement
```

**Parameters:**

|Parameter|Type|Default|Description|
|---|---|---|---|
|`voltage_trace`|WaveformTrace|required|Voltage waveform across capacitor|
|`current_trace`|WaveformTrace \|None|`None`|Current waveform (required for some methods)|
|`method`|str|`"charge"`|Measurement method (see below)|
|`resistance`|float \|None|`None`|Known resistance for frequency method|

**Methods:**

- `"charge"`: C = Q/V = integral(I·dt) / ΔV (most accurate, requires current)
- `"slope"`: C = I / (dV/dt) (requires current)
- `"frequency"`: Extract from RC time constant (requires resistance)

**Returns:**

- `CapacitanceMeasurement`: Measurement result object

**CapacitanceMeasurement Attributes:**

|Attribute|Type|Description|
|---|---|---|
|`capacitance`|float|Measured capacitance in Farads|
|`esr`|float|Equivalent Series Resistance (ohms)|
|`method`|str|Measurement method used|
|`confidence`|float|Confidence in measurement (0-1)|
|`statistics`|dict|Additional measurement statistics|

**Example:**

```python
import oscura as osc

# Method 1: Charge integration (most accurate)
voltage = osc.load("cap_voltage.wfm")
current = osc.load("cap_current.wfm")

C_result = osc.measure_capacitance(voltage, current, method="charge")
print(f"Capacitance: {C_result.capacitance * 1e12:.2f} pF")
print(f"ESR: {C_result.esr:.2f} ohms")

# Method 2: Slope method
C_result = osc.measure_capacitance(voltage, current, method="slope")
print(f"Capacitance: {C_result.capacitance * 1e12:.2f} pF")

# Method 3: Time constant (no current trace needed)
C_result = osc.measure_capacitance(
    voltage,
    method="frequency",
    resistance=1000.0  # 1 kOhm
)
print(f"Capacitance: {C_result.capacitance * 1e9:.2f} nF")
print(f"Time constant: {C_result.statistics['time_constant'] * 1e6:.2f} us")
```

**Best Practices:**

- Use **"charge"** method for most accurate results with clean I/V data
- Use **"slope"** method for noisy measurements (more robust)
- Use **"frequency"** method when only voltage is available
- Ensure adequate sampling rate (at least 10x the signal frequency)
- For AC measurements, measure at known frequency for best accuracy

**References:**

- COMP-002: Capacitance Measurement Specification

---

### osc.measure_inductance()

Measure inductance from voltage/current waveforms using the relationship V = L·dI/dt.

**Signature:**

```python
osc.measure_inductance(
    voltage_trace,
    current_trace=None,
    *,
    method="slope",
    resistance=None
) -> InductanceMeasurement
```

**Parameters:**

|Parameter|Type|Default|Description|
|---|---|---|---|
|`voltage_trace`|WaveformTrace|required|Voltage waveform across inductor|
|`current_trace`|WaveformTrace \|None|`None`|Current waveform (required for some methods)|
|`method`|str|`"slope"`|Measurement method (see below)|
|`resistance`|float \|None|`None`|Known resistance for frequency method|

**Methods:**

- `"flux"`: L = Φ/I = integral(V·dt) / ΔI (most accurate, requires current)
- `"slope"`: L = V / (dI/dt) (default, requires current)
- `"frequency"`: Extract from RL time constant (requires resistance)

**Returns:**

- `InductanceMeasurement`: Measurement result object

**InductanceMeasurement Attributes:**

|Attribute|Type|Description|
|---|---|---|
|`inductance`|float|Measured inductance in Henrys|
|`dcr`|float|DC Resistance in ohms|
|`q_factor`|float \|None|Quality factor at measurement frequency|
|`method`|str|Measurement method used|
|`confidence`|float|Confidence in measurement (0-1)|
|`statistics`|dict|Additional measurement statistics|

**Example:**

```python
import oscura as osc

# Method 1: Slope method (default)
voltage = osc.load("inductor_voltage.wfm")
current = osc.load("inductor_current.wfm")

L_result = osc.measure_inductance(voltage, current, method="slope")
print(f"Inductance: {L_result.inductance * 1e6:.2f} uH")
print(f"DCR: {L_result.dcr:.3f} ohms")

# Method 2: Flux integration (most accurate)
L_result = osc.measure_inductance(voltage, current, method="flux")
print(f"Inductance: {L_result.inductance * 1e6:.2f} uH")
print(f"Confidence: {L_result.confidence:.1%}")

# Method 3: Time constant
L_result = osc.measure_inductance(
    voltage,
    method="frequency",
    resistance=10.0  # 10 ohm series resistance
)
print(f"Inductance: {L_result.inductance * 1e6:.2f} uH")
print(f"Time constant: {L_result.statistics['time_constant'] * 1e6:.2f} us")
```

**Best Practices:**

- Use **"flux"** method for most accurate DC measurements
- Use **"slope"** method for transient analysis
- Ensure sufficient current change (ΔI) for accurate measurements
- Account for DCR in precision measurements
- For high-frequency inductors, use impedance analyzer instead

**References:**

- COMP-003: Inductance Measurement Specification

---

### osc.extract_parasitics()

Extract parasitic R, L, C parameters from impedance measurements by fitting an equivalent circuit model.

**Signature:**

```python
osc.extract_parasitics(
    voltage_trace,
    current_trace,
    *,
    model="series_RLC",
    frequency_range=None
) -> ParasiticExtraction
```

**Parameters:**

|Parameter|Type|Default|Description|
|---|---|---|---|
|`voltage_trace`|WaveformTrace|required|Voltage waveform|
|`current_trace`|WaveformTrace|required|Current waveform|
|`model`|str|`"series_RLC"`|Equivalent circuit model type|
|`frequency_range`|tuple[float, float] \|None|`None`|Frequency range for analysis (Hz)|

**Models:**

- `"series_RLC"`: Series R-L-C equivalent circuit
- `"parallel_RLC"`: Parallel R-L-C equivalent circuit

**Returns:**

- `ParasiticExtraction`: Extraction result object

**ParasiticExtraction Attributes:**

|Attribute|Type|Description|
|---|---|---|
|`capacitance`|float|Parasitic capacitance in Farads|
|`inductance`|float|Parasitic inductance in Henrys|
|`resistance`|float|Parasitic resistance in ohms|
|`model_type`|str|Equivalent circuit model used|
|`resonant_freq`|float \|None|Self-resonant frequency (Hz)|
|`fit_quality`|float|Quality of model fit (R-squared, 0-1)|

**Example:**

```python
import oscura as osc

# Extract parasitics from impedance sweep
voltage = osc.load("impedance_voltage.wfm")
current = osc.load("impedance_current.wfm")

# Series RLC model
params = osc.extract_parasitics(
    voltage,
    current,
    model="series_RLC"
)

print(f"Series RLC Model:")
print(f"  R = {params.resistance:.3f} ohms")
print(f"  L = {params.inductance * 1e9:.2f} nH")
print(f"  C = {params.capacitance * 1e12:.2f} pF")

if params.resonant_freq:
    print(f"  Self-resonant freq: {params.resonant_freq / 1e6:.2f} MHz")

print(f"  Fit quality (R²): {params.fit_quality:.3f}")

# Parallel RLC model
params_par = osc.extract_parasitics(
    voltage,
    current,
    model="parallel_RLC",
    frequency_range=(1e6, 100e6)  # 1-100 MHz
)

print(f"\nParallel RLC Model:")
print(f"  R = {params_par.resistance:.3f} ohms")
print(f"  L = {params_par.inductance * 1e6:.2f} uH")
print(f"  C = {params_par.capacitance * 1e12:.2f} pF")
```

**Use Cases:**

- **Component Modeling** - Create SPICE models from measurements
- **Capacitor ESL/ESR** - Extract parasitic inductance and resistance
- **Inductor SRF** - Determine self-resonant frequency
- **PCB Parasitics** - Model parasitic components in traces
- **Power Distribution** - Characterize power plane impedance

**Best Practices:**

- Use wide frequency range for accurate extraction
- Verify fit quality (R² > 0.9 is good)
- Compare both series and parallel models
- Account for measurement system impedance
- Validate results with known components

**References:**

- COMP-004: Parasitic Extraction Specification

---

## Transmission Line Analysis

### osc.characteristic_impedance()

Extract characteristic impedance from TDR measurement. Calculates Z0 from a stable region of the TDR waveform.

**Signature:**

```python
osc.characteristic_impedance(
    trace,
    *,
    z0_source=50.0,
    start_time=None,
    end_time=None
) -> float
```

**Parameters:**

|Parameter|Type|Default|Description|
|---|---|---|---|
|`trace`|WaveformTrace|required|TDR reflection waveform|
|`z0_source`|float|`50.0`|Source impedance (ohms)|
|`start_time`|float \|None|`None`|Start of analysis window (seconds)|
|`end_time`|float \|None|`None`|End of analysis window (seconds)|

**Returns:**

- `float`: Characteristic impedance in ohms

**Example:**

```python
import oscura as osc

# Simple impedance extraction
tdr_trace = osc.load("tdr_measurement.wfm")
z0 = osc.characteristic_impedance(tdr_trace, z0_source=50.0)

print(f"Characteristic impedance: {z0:.2f} ohms")

# Analyze specific region (avoid reflections)
z0 = osc.characteristic_impedance(
    tdr_trace,
    z0_source=50.0,
    start_time=2e-9,   # Start at 2 ns
    end_time=8e-9      # End at 8 ns
)

print(f"Z0 (stable region): {z0:.2f} ohms")

# Verification for 50-ohm line
tolerance = abs(z0 - 50.0)
if tolerance < 5.0:
    print(f"✓ Within spec (±5 ohms)")
else:
    print(f"✗ Out of spec: {tolerance:.1f} ohms deviation")
```

---

### osc.propagation_delay()

Measure propagation delay from TDR waveform. Calculates the one-way time for a signal to travel down the line.

**Signature:**

```python
osc.propagation_delay(
    trace,
    *,
    threshold=0.5
) -> float
```

**Parameters:**

|Parameter|Type|Default|Description|
|---|---|---|---|
|`trace`|WaveformTrace|required|TDR reflection waveform|
|`threshold`|float|`0.5`|Threshold level for edge detection (normalized)|

**Returns:**

- `float`: Propagation delay in seconds

**Example:**

```python
import oscura as osc

# Measure propagation delay
tdr_trace = osc.load("tdr_trace.wfm")
delay = osc.propagation_delay(tdr_trace)

print(f"Propagation delay: {delay * 1e9:.2f} ns")
print(f"Propagation delay: {delay * 1e12:.0f} ps")

# Calculate from known length and velocity
line_length = 0.100  # 100 mm = 0.1 m
velocity_factor = 0.66  # FR4
c = 299792458  # Speed of light (m/s)

theoretical_delay = line_length / (c * velocity_factor)
print(f"Theoretical delay: {theoretical_delay * 1e9:.2f} ns")
print(f"Measured vs theoretical: {(delay / theoretical_delay - 1) * 100:+.1f}%")

# Use custom threshold for noisy signals
delay = osc.propagation_delay(tdr_trace, threshold=0.3)
```

---

### osc.velocity_factor()

Calculate velocity factor from TDR and known line length. Determines how fast signals propagate as a fraction of the speed of light.

**Signature:**

```python
osc.velocity_factor(
    trace,
    line_length
) -> float
```

**Parameters:**

|Parameter|Type|Default|Description|
|---|---|---|---|
|`trace`|WaveformTrace|required|TDR reflection waveform|
|`line_length`|float|required|Known line length in meters|

**Returns:**

- `float`: Velocity factor (0 to 1)

**Example:**

```python
import oscura as osc
import numpy as np

# Measure velocity factor with known length
tdr_trace = osc.load("tdr_trace.wfm")
line_length = 0.050  # 50 mm trace

vf = osc.velocity_factor(tdr_trace, line_length)
print(f"Velocity factor: {vf:.3f}")

# Calculate effective dielectric constant
epsilon_eff = 1 / (vf ** 2)
print(f"Effective dielectric constant: {epsilon_eff:.2f}")

# Compare with common materials
materials = {
    "Air/Vacuum": 1.00,
    "PTFE (Teflon)": 2.1,
    "FR4": 4.4,
    "Rogers 4003C": 3.38,
    "Rogers 4350B": 3.48,
    "Polyimide": 3.5,
}

print("\nMaterial comparison:")
for material, er in materials.items():
    vf_material = 1 / np.sqrt(er)
    diff = abs(vf - vf_material)
    print(f"  {material:20s}: VF={vf_material:.3f}, Er={er:.2f}, diff={diff:.3f}")
```

**Common Velocity Factors:**

|Material|Typical Er|Velocity Factor|
|---|---|---|
|Air/Vacuum|1.00|1.000|
|PTFE|2.10|0.690|
|Rogers 4003C|3.38|0.544|
|Rogers 4350B|3.48|0.536|
|Polyimide|3.50|0.535|
|FR4|4.40|0.476|

---

## Advanced Usage

### Complete TDR Analysis

Perform comprehensive transmission line characterization:

```python
import oscura as osc
import matplotlib.pyplot as plt

# Load TDR measurement
tdr_trace = osc.load("pcb_trace_tdr.wfm")

# Extract all parameters
z0, profile = osc.extract_impedance(tdr_trace, z0_source=50.0)
delay = osc.propagation_delay(tdr_trace)
vf = osc.velocity_factor(tdr_trace, line_length=0.075)  # 75mm trace

# Find discontinuities
discontinuities = osc.discontinuity_analysis(tdr_trace, threshold=3.0)

# Generate report
print("=== TDR Analysis Report ===\n")
print(f"Characteristic Impedance: {z0:.2f} ohms")
print(f"Propagation Delay: {delay * 1e9:.2f} ns")
print(f"Velocity Factor: {vf:.3f}")
print(f"Effective Er: {(1/vf)**2:.2f}")
print(f"\nImpedance Statistics:")
print(f"  Mean: {profile.mean_impedance:.2f} ohms")
print(f"  Min:  {profile.min_impedance:.2f} ohms")
print(f"  Max:  {profile.max_impedance:.2f} ohms")
print(f"  Std:  {profile.statistics['z0_std']:.2f} ohms")

print(f"\nDiscontinuities: {len(discontinuities)}")
for i, disc in enumerate(discontinuities, 1):
    print(f"  {i}. @ {disc.position*1000:.1f}mm: {disc.magnitude:+.1f} ohms ({disc.discontinuity_type})")

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot impedance profile
ax1.plot(profile.distance * 1000, profile.impedance, 'b-', linewidth=1.5)
ax1.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Target 50Ω')
ax1.fill_between(profile.distance * 1000, 45, 55, alpha=0.2, color='green', label='±10% tolerance')
ax1.set_xlabel('Distance (mm)')
ax1.set_ylabel('Impedance (Ω)')
ax1.set_title('TDR Impedance Profile')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Mark discontinuities
for disc in discontinuities:
    ax1.axvline(x=disc.position * 1000, color='orange', linestyle=':', alpha=0.7)
    ax1.annotate(f'{disc.magnitude:+.0f}Ω',
                xy=(disc.position * 1000, disc.impedance_after),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, color='orange')

# Plot raw TDR waveform
time_axis = profile.time * 1e9  # Convert to ns
ax2.plot(time_axis, tdr_trace.data, 'g-', linewidth=1)
ax2.set_xlabel('Time (ns)')
ax2.set_ylabel('Voltage (V)')
ax2.set_title('TDR Waveform')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tdr_analysis.png', dpi=300)
plt.show()
```

### Component Parasitic Analysis

Extract and compare parasitic models:

```python
import oscura as osc
import matplotlib.pyplot as plt
import numpy as np

# Load impedance measurement sweep
voltage = osc.load("component_voltage.wfm")
current = osc.load("component_current.wfm")

# Extract both series and parallel models
series_params = osc.extract_parasitics(voltage, current, model="series_RLC")
parallel_params = osc.extract_parasitics(voltage, current, model="parallel_RLC")

# Display results
print("=== Parasitic Extraction Results ===\n")

print("Series RLC Model:")
print(f"  R = {series_params.resistance:.3f} Ω")
print(f"  L = {series_params.inductance * 1e9:.2f} nH")
print(f"  C = {series_params.capacitance * 1e12:.2f} pF")
print(f"  SRF = {series_params.resonant_freq / 1e6:.2f} MHz" if series_params.resonant_freq else "  SRF = N/A")
print(f"  Fit quality: {series_params.fit_quality:.3f}")

print("\nParallel RLC Model:")
print(f"  R = {parallel_params.resistance:.3f} Ω")
print(f"  L = {parallel_params.inductance * 1e9:.2f} nH")
print(f"  C = {parallel_params.capacitance * 1e12:.2f} pF")
print(f"  SRF = {parallel_params.resonant_freq / 1e6:.2f} MHz" if parallel_params.resonant_freq else "  SRF = N/A")
print(f"  Fit quality: {parallel_params.fit_quality:.3f}")

# Calculate impedance magnitude vs frequency
freqs = np.logspace(3, 9, 1000)  # 1 kHz to 1 GHz
omega = 2 * np.pi * freqs

# Series RLC impedance
R_s, L_s, C_s = series_params.resistance, series_params.inductance, series_params.capacitance
Z_series = np.abs(R_s + 1j * (omega * L_s - 1 / (omega * C_s)))

# Parallel RLC impedance
R_p, L_p, C_p = parallel_params.resistance, parallel_params.inductance, parallel_params.capacitance
Y_parallel = 1/R_p + 1j * omega * C_p + 1 / (1j * omega * L_p)
Z_parallel = np.abs(1 / Y_parallel)

# Plot
plt.figure(figsize=(12, 6))
plt.loglog(freqs / 1e6, Z_series, 'b-', label='Series RLC', linewidth=2)
plt.loglog(freqs / 1e6, Z_parallel, 'r--', label='Parallel RLC', linewidth=2)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Impedance Magnitude (Ω)')
plt.title('Parasitic Impedance Models')
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.savefig('parasitic_impedance.png', dpi=300)
plt.show()
```

### Quality Control Automation

Automate impedance verification for manufacturing:

```python
import oscura as osc
from dataclasses import dataclass

@dataclass
class ImpedanceSpec:
    """Impedance specification for QC."""
    target: float
    tolerance: float  # ±ohms
    min_length: float  # meters
    max_discontinuity: float  # ohms

def verify_impedance_compliance(tdr_trace, spec: ImpedanceSpec) -> dict:
    """Verify TDR measurement against specification."""

    # Extract impedance
    z0, profile = osc.extract_impedance(tdr_trace)

    # Check impedance tolerance
    impedance_dev = abs(z0 - spec.target)
    impedance_pass = impedance_dev <= spec.tolerance

    # Check impedance uniformity
    impedance_std = profile.statistics['z0_std']
    uniformity_pass = impedance_std < spec.tolerance / 2

    # Check for discontinuities
    discontinuities = osc.discontinuity_analysis(
        tdr_trace,
        threshold=spec.max_discontinuity
    )
    discontinuity_pass = len(discontinuities) == 0

    # Overall pass/fail
    passed = impedance_pass and uniformity_pass and discontinuity_pass

    return {
        'passed': passed,
        'z0_measured': z0,
        'z0_deviation': impedance_dev,
        'z0_std': impedance_std,
        'impedance_pass': impedance_pass,
        'uniformity_pass': uniformity_pass,
        'discontinuity_count': len(discontinuities),
        'discontinuity_pass': discontinuity_pass,
        'discontinuities': discontinuities,
    }

# Define specification
spec = ImpedanceSpec(
    target=50.0,          # 50 ohm target
    tolerance=5.0,        # ±5 ohm tolerance
    min_length=0.010,     # 10mm minimum
    max_discontinuity=3.0 # Max 3 ohm discontinuity
)

# Test multiple traces
traces = [
    "trace_1.wfm",
    "trace_2.wfm",
    "trace_3.wfm",
]

print("=== Impedance QC Report ===\n")
results = []

for trace_file in traces:
    tdr_trace = osc.load(trace_file)
    result = verify_impedance_compliance(tdr_trace, spec)
    results.append((trace_file, result))

    status = "✓ PASS" if result['passed'] else "✗ FAIL"
    print(f"{trace_file}: {status}")
    print(f"  Z0: {result['z0_measured']:.2f} Ω (dev: {result['z0_deviation']:.2f} Ω)")
    print(f"  Std: {result['z0_std']:.2f} Ω")
    print(f"  Discontinuities: {result['discontinuity_count']}")

    if not result['passed']:
        if not result['impedance_pass']:
            print(f"  ⚠ Impedance out of spec")
        if not result['uniformity_pass']:
            print(f"  ⚠ Poor uniformity")
        if not result['discontinuity_pass']:
            print(f"  ⚠ Discontinuities detected")
    print()

# Summary
passed_count = sum(1 for _, r in results if r['passed'])
print(f"Summary: {passed_count}/{len(results)} traces passed")
```

## Integration with Analysis Pipeline

Component analysis integrates seamlessly with Oscura's other analysis capabilities:

```python
import oscura as osc
from oscura.reporting import generate_report, save_pdf_report

# Load and analyze
tdr_trace = osc.load("measurement.wfm")

# Component analysis
z0, profile = osc.extract_impedance(tdr_trace)
delay = osc.propagation_delay(tdr_trace)
discontinuities = osc.discontinuity_analysis(tdr_trace)

# Combine with other analyses
freq = osc.frequency(tdr_trace)
rise_time = osc.rise_time(tdr_trace)

# Generate comprehensive report
report = generate_report(
    tdr_trace,
    title="TDR Analysis Report",
    include_plots=True
)

# Add custom component analysis section
report['component_analysis'] = {
    'characteristic_impedance': z0,
    'propagation_delay': delay,
    'discontinuity_count': len(discontinuities),
    'impedance_profile': profile,
}

# Save report
save_pdf_report(report, "tdr_report.pdf")
```

## Error Handling

Component analysis functions raise specific exceptions for error conditions:

```python
import oscura as osc
from oscura.core.exceptions import InsufficientDataError, AnalysisError

try:
    z0, profile = osc.extract_impedance(tdr_trace)
except InsufficientDataError as e:
    print(f"Not enough data: {e.message}")
    print(f"Required: {e.required} samples, Available: {e.available}")
except AnalysisError as e:
    print(f"Analysis failed: {e.message}")

# Check data quality before measurement
if len(tdr_trace.data) < 100:
    print("Warning: Short trace may produce unreliable results")

# Validate results
if z0 < 1.0 or z0 > 1000.0:
    print(f"Warning: Impedance {z0:.1f}Ω is out of reasonable range")
```

## Best Practices

### TDR Measurements

1. **Calibration** - Always calibrate your TDR system with known standards
2. **Source Impedance** - Specify correct z0_source matching your equipment
3. **Rise Time** - Use fast edge rates (<100ps) for accurate measurements
4. **Sampling Rate** - Ensure 10× oversampling of the fastest edge
5. **Analysis Window** - Choose stable regions avoiding initial edge and end reflections

### Component Measurements

1. **Synchronization** - Ensure voltage and current traces are time-aligned
2. **Bandwidth** - Use adequate measurement bandwidth for accuracy
3. **Loading Effects** - Account for probe and fixture parasitics
4. **Method Selection** - Choose appropriate method based on signal characteristics
5. **Validation** - Cross-check results with known reference components

### Parasitic Extraction

1. **Frequency Range** - Use wide frequency range covering resonances
2. **Model Selection** - Try both series and parallel models
3. **Fit Quality** - Verify R² > 0.9 for reliable results
4. **Physical Validation** - Ensure extracted values are physically reasonable
5. **Measurement System** - Calibrate out fixture and cable effects

## See Also

- [Analysis API](analysis.md) - General signal analysis functions
- [Loader API](loader.md) - Loading waveform data
- [Export API](export.md) - Exporting measurement data
- [Reporting API](reporting.md) - Generating analysis reports
- [IPC-TM-650 2.5.5.7](https://www.ipc.org/) - TDR impedance standards
- [IEEE 370-2020](https://standards.ieee.org/) - Electrical characterization of interconnects
