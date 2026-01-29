# Oscura Power Analysis API Documentation

> **Version**: 0.6.0 | **Last Updated**: 2026-01-25

Complete guide to power analysis functions for DC, AC, switching, and efficiency measurements.

## Overview

Oscura provides comprehensive power analysis functionality for electrical systems:

- **Basic Power Analysis** - Instantaneous, average, RMS, peak power, and energy
- **AC Power Analysis** - Apparent power, reactive power, and power factor
- **Efficiency Analysis** - Power conversion efficiency and loss breakdown
- **Ripple Analysis** - AC ripple measurement on DC signals
- **Statistics** - Comprehensive power statistics and profiling

All power analysis functions are accessible via top-level API: `osc.instantaneous_power()`, `osc.power_factor()`, etc.

## Quick Start

```python
import oscura as osc

# Load voltage and current traces
v_trace = osc.load("voltage.wfm")
i_trace = osc.load("current.wfm")

# Basic power analysis
power = osc.instantaneous_power(v_trace, i_trace)
p_avg = osc.average_power(power)
e_total = osc.energy(power)

# AC power analysis
s = osc.apparent_power(v_trace, i_trace)
q = osc.reactive_power(v_trace, i_trace)
pf = osc.power_factor(v_trace, i_trace)

# Efficiency analysis
eta = osc.efficiency(v_in, i_in, v_out, i_out)

# Ripple analysis
r_pp, r_rms = osc.ripple(dc_output)
stats = osc.ripple_statistics(dc_output)
```

## Basic Power Analysis

### instantaneous_power()

Calculate instantaneous power from voltage and current traces.

#### Function Signature

```python
osc.instantaneous_power(
    voltage,                    # Voltage waveform trace
    current,                    # Current waveform trace
    *,
    interpolate_if_needed=True  # Auto-interpolate if sample rates differ
) -> WaveformTrace
```

#### Parameters

- **voltage** (`WaveformTrace`) - Voltage waveform trace
- **current** (`WaveformTrace`) - Current waveform trace
- **interpolate_if_needed** (`bool`, optional) - If `True` (default), automatically interpolate if voltage and current have different sample rates. If `False`, raises `AnalysisError` on sample rate mismatch.

#### Returns

`WaveformTrace` - Power waveform trace where `P(t) = V(t) × I(t)`. Units are Watts if inputs are in Volts and Amperes.

#### Examples

```python
import oscura as osc

# Load voltage and current
v = osc.load("voltage.wfm")
i = osc.load("current.wfm")

# Calculate instantaneous power
power = osc.instantaneous_power(v, i)
print(f"Peak power: {max(power.data):.2f} W")

# Disable auto-interpolation (strict mode)
try:
    power = osc.instantaneous_power(v, i, interpolate_if_needed=False)
except osc.AnalysisError as e:
    print(f"Sample rate mismatch: {e}")
```

#### Notes

- Automatically handles different sample rates by interpolating to the higher rate
- Handles different trace lengths by truncating to the shorter length
- Uses linear interpolation when sample rate adjustment is needed
- Based on IEC 61000-4-7 power measurement standards

### average_power()

Calculate average (mean) power over the entire trace duration.

#### Function Signature

```python
osc.average_power(
    power=None,      # Power trace (if already calculated)
    *,
    voltage=None,    # Voltage trace (alternative)
    current=None     # Current trace (alternative)
) -> float
```

#### Parameters

- **power** (`WaveformTrace`, optional) - Pre-calculated power trace
- **voltage** (`WaveformTrace`, optional) - Voltage trace (used with `current`)
- **current** (`WaveformTrace`, optional) - Current trace (used with `voltage`)

Must provide either `power` OR both `voltage` and `current`.

#### Returns

`float` - Average power in Watts. Calculated as `P_avg = (1/T) ∫ P(t) dt`

#### Examples

```python
import oscura as osc

# Method 1: From power trace
power = osc.instantaneous_power(v, i)
p_avg = osc.average_power(power)

# Method 2: Directly from voltage and current
p_avg = osc.average_power(voltage=v, current=i)

print(f"Average power: {p_avg:.2f} W")
```

### energy()

Calculate total energy consumption (integral of power over time).

#### Function Signature

```python
osc.energy(
    power=None,        # Power trace
    *,
    voltage=None,      # Voltage trace (alternative)
    current=None,      # Current trace (alternative)
    start_time=None,   # Start time for integration (seconds)
    end_time=None      # End time for integration (seconds)
) -> float
```

#### Parameters

- **power** (`WaveformTrace`, optional) - Pre-calculated power trace
- **voltage** (`WaveformTrace`, optional) - Voltage trace
- **current** (`WaveformTrace`, optional) - Current trace
- **start_time** (`float`, optional) - Start time for integration in seconds
- **end_time** (`float`, optional) - End time for integration in seconds

#### Returns

`float` - Energy in Joules. Calculated as `E = ∫ P(t) dt`

#### Examples

```python
import oscura as osc

# Total energy over entire trace
e_total = osc.energy(voltage=v, current=i)
print(f"Total energy: {e_total*1e3:.2f} mJ")

# Energy over specific time window
e_window = osc.energy(
    voltage=v,
    current=i,
    start_time=1e-6,  # 1 µs
    end_time=10e-6     # 10 µs
)
print(f"Energy (1-10 µs): {e_window*1e6:.2f} µJ")

# From pre-calculated power trace
power = osc.instantaneous_power(v, i)
e = osc.energy(power)
```

#### Notes

- Uses trapezoidal integration for accurate results
- Time limits apply to the trace's time axis starting from t=0
- Useful for battery life estimation and power budgeting

### power_statistics()

Calculate comprehensive power statistics including average, RMS, peak, and energy.

#### Function Signature

```python
osc.power_statistics(
    power=None,      # Power trace
    *,
    voltage=None,    # Voltage trace (alternative)
    current=None     # Current trace (alternative)
) -> dict[str, float]
```

#### Parameters

- **power** (`WaveformTrace`, optional) - Pre-calculated power trace
- **voltage** (`WaveformTrace`, optional) - Voltage trace
- **current** (`WaveformTrace`, optional) - Current trace

#### Returns

`dict[str, float]` - Dictionary containing:

- **average** - Mean power (W)
- **rms** - RMS power (W)
- **peak** - Peak absolute power (W)
- **peak_positive** - Maximum positive power (W)
- **peak_negative** - Minimum power, for regenerative loads (W)
- **energy** - Total energy (J)
- **min** - Minimum power value (W)
- **std** - Standard deviation of power (W)

#### Examples

```python
import oscura as osc

# Get all power statistics
stats = osc.power_statistics(voltage=v, current=i)

print(f"Average power:  {stats['average']:.2f} W")
print(f"RMS power:      {stats['rms']:.2f} W")
print(f"Peak power:     {stats['peak']:.2f} W")
print(f"Total energy:   {stats['energy']*1e3:.2f} mJ")
print(f"Std deviation:  {stats['std']:.2f} W")

# Check for regenerative braking
if stats['peak_negative'] < 0:
    print(f"Regeneration detected: {abs(stats['peak_negative']):.2f} W")
```

## AC Power Analysis

### apparent_power()

Calculate apparent power (S) for AC circuits.

#### Function Signature

```python
osc.apparent_power(
    voltage,    # Voltage waveform trace
    current     # Current waveform trace
) -> float
```

#### Parameters

- **voltage** (`WaveformTrace`) - AC voltage waveform
- **current** (`WaveformTrace`) - AC current waveform

#### Returns

`float` - Apparent power in VA (volt-amperes). Calculated as `S = V_rms × I_rms`

#### Examples

```python
import oscura as osc

# Load AC signals
v_ac = osc.load("ac_voltage.wfm")
i_ac = osc.load("ac_current.wfm")

# Calculate apparent power
s = osc.apparent_power(v_ac, i_ac)
print(f"Apparent power: {s:.2f} VA")

# Compare with real power
p = osc.average_power(voltage=v_ac, current=i_ac)
print(f"Real power:     {p:.2f} W")
print(f"Difference:     {s-p:.2f} VAR (reactive)")
```

#### References

- IEEE 1459-2010: Standard Definitions for the Measurement of Electric Power Quantities

### reactive_power()

Calculate reactive power (Q) for AC circuits.

#### Function Signature

```python
osc.reactive_power(
    voltage,           # Voltage waveform trace
    current,           # Current waveform trace
    *,
    frequency=None     # Fundamental frequency (Hz), auto-detect if None
) -> float
```

#### Parameters

- **voltage** (`WaveformTrace`) - AC voltage waveform
- **current** (`WaveformTrace`) - AC current waveform
- **frequency** (`float`, optional) - Fundamental frequency in Hz. If `None`, automatically detected.

#### Returns

`float` - Reactive power in VAR (volt-amperes reactive). Positive for inductive loads, negative for capacitive loads. Calculated as `Q = V_rms × I_rms × sin(φ)`

#### Examples

```python
import oscura as osc

# Auto-detect frequency
q = osc.reactive_power(v_ac, i_ac)
print(f"Reactive power: {q:.2f} VAR")

if q > 0:
    print("Inductive load (current lags voltage)")
elif q < 0:
    print("Capacitive load (current leads voltage)")

# Specify fundamental frequency
q = osc.reactive_power(v_ac, i_ac, frequency=60.0)

# Calculate power triangle
p = osc.average_power(voltage=v_ac, current=i_ac)
s = osc.apparent_power(v_ac, i_ac)
print(f"Real power:     {p:.2f} W")
print(f"Reactive power: {q:.2f} VAR")
print(f"Apparent power: {s:.2f} VA")
```

#### Notes

- Uses cross-correlation to determine phase angle between voltage and current
- Positive reactive power indicates inductive load (motor-like)
- Negative reactive power indicates capacitive load

### power_factor()

Calculate power factor (PF) for AC circuits.

#### Function Signature

```python
osc.power_factor(
    voltage,    # Voltage waveform trace
    current     # Current waveform trace
) -> float
```

#### Parameters

- **voltage** (`WaveformTrace`) - AC voltage waveform
- **current** (`WaveformTrace`) - AC current waveform

#### Returns

`float` - Power factor (dimensionless, range 0 to 1). Calculated as `PF = P / S`

For sinusoidal waveforms, `PF = cos(φ)`. For non-sinusoidal waveforms, includes distortion effects.

#### Examples

```python
import oscura as osc

# Calculate power factor
pf = osc.power_factor(v_ac, i_ac)
print(f"Power factor: {pf:.3f}")

# Interpret the result
if pf > 0.95:
    print("Excellent power factor")
elif pf > 0.85:
    print("Good power factor")
elif pf > 0.7:
    print("Fair power factor - consider correction")
else:
    print("Poor power factor - correction recommended")

# Calculate all AC power quantities
p = osc.average_power(voltage=v_ac, current=i_ac)
s = osc.apparent_power(v_ac, i_ac)
q = osc.reactive_power(v_ac, i_ac)
pf = osc.power_factor(v_ac, i_ac)

print(f"\nAC Power Analysis:")
print(f"  Real Power (P):     {p:.2f} W")
print(f"  Reactive Power (Q): {q:.2f} VAR")
print(f"  Apparent Power (S): {s:.2f} VA")
print(f"  Power Factor:       {pf:.3f}")
print(f"  Phase angle:        {np.degrees(np.arccos(pf)):.1f}°")
```

#### Notes

- Power factor of 1.0 indicates purely resistive load (ideal)
- Power factor < 1.0 indicates reactive component (inductance or capacitance)
- Can be negative for regenerative loads
- Based on IEEE 1459-2010 standard definitions

## Efficiency Analysis

### efficiency()

Calculate power conversion efficiency.

#### Function Signature

```python
osc.efficiency(
    v_in,     # Input voltage trace
    i_in,     # Input current trace
    v_out,    # Output voltage trace
    i_out     # Output current trace
) -> float
```

#### Parameters

- **v_in** (`WaveformTrace`) - Input voltage waveform
- **i_in** (`WaveformTrace`) - Input current waveform
- **v_out** (`WaveformTrace`) - Output voltage waveform
- **i_out** (`WaveformTrace`) - Output current waveform

#### Returns

`float` - Efficiency as a ratio (0 to 1). Calculated as `η = P_out / P_in`

#### Examples

```python
import oscura as osc

# Load input and output traces
v_in = osc.load("input_voltage.wfm")
i_in = osc.load("input_current.wfm")
v_out = osc.load("output_voltage.wfm")
i_out = osc.load("output_current.wfm")

# Calculate efficiency
eta = osc.efficiency(v_in, i_in, v_out, i_out)
print(f"Efficiency: {eta*100:.1f}%")

# Calculate power budget
p_in = osc.average_power(voltage=v_in, current=i_in)
p_out = osc.average_power(voltage=v_out, current=i_out)
losses = p_in - p_out

print(f"\nPower Budget:")
print(f"  Input:      {p_in:.2f} W")
print(f"  Output:     {p_out:.2f} W")
print(f"  Losses:     {losses:.2f} W ({losses/p_in*100:.1f}%)")
print(f"  Efficiency: {eta*100:.2f}%")
```

#### Use Cases

- DC-DC converter efficiency measurement
- Power supply characterization
- Motor drive efficiency analysis
- Battery charger performance testing

## Ripple Analysis

### ripple()

Measure AC ripple on a DC signal.

#### Function Signature

```python
osc.ripple(
    trace,             # DC waveform with AC ripple
    *,
    dc_coupling=False  # Include DC component in measurement
) -> tuple[float, float]
```

#### Parameters

- **trace** (`WaveformTrace`) - DC voltage or current waveform with AC ripple
- **dc_coupling** (`bool`, optional) - If `False` (default), removes DC component for pure AC ripple measurement. If `True`, includes DC component.

#### Returns

`tuple[float, float]` - Tuple of `(ripple_pp, ripple_rms)`:

- **ripple_pp** - Peak-to-peak ripple amplitude in signal units
- **ripple_rms** - RMS ripple amplitude in signal units

#### Examples

```python
import oscura as osc

# Load DC output with ripple
dc_out = osc.load("dc_output.wfm")

# Measure ripple
r_pp, r_rms = osc.ripple(dc_out)
print(f"Ripple: {r_pp*1e3:.2f} mV pk-pk")
print(f"        {r_rms*1e3:.2f} mV RMS")

# Measure ripple as percentage of DC level
dc_level = osc.mean(dc_out)
print(f"DC level: {dc_level:.3f} V")
print(f"Ripple:   {r_pp/dc_level*100:.3f}% pk-pk")
print(f"          {r_rms/dc_level*100:.3f}% RMS")
```

#### Notes

- Default behavior (dc_coupling=False) removes DC component for pure ripple
- Based on IEC 61000-4-7 power quality standards
- Useful for power supply output quality assessment

### ripple_statistics()

Calculate comprehensive ripple statistics.

#### Function Signature

```python
osc.ripple_statistics(
    trace    # DC waveform with AC ripple
) -> dict[str, float]
```

#### Parameters

- **trace** (`WaveformTrace`) - DC voltage or current waveform with AC ripple

#### Returns

`dict[str, float]` - Dictionary containing:

- **dc_level** - DC (mean) level
- **ripple_pp** - Peak-to-peak ripple amplitude
- **ripple_rms** - RMS ripple amplitude
- **ripple_pp_percent** - Peak-to-peak as percentage of DC level
- **ripple_rms_percent** - RMS as percentage of DC level
- **ripple_frequency** - Dominant ripple frequency (Hz)
- **crest_factor** - Ripple peak / ripple RMS ratio

#### Examples

```python
import oscura as osc

# Get comprehensive ripple analysis
stats = osc.ripple_statistics(dc_output)

print(f"DC Level:         {stats['dc_level']:.3f} V")
print(f"Ripple (pk-pk):   {stats['ripple_pp']*1e3:.2f} mV ({stats['ripple_pp_percent']:.3f}%)")
print(f"Ripple (RMS):     {stats['ripple_rms']*1e3:.2f} mV ({stats['ripple_rms_percent']:.3f}%)")
print(f"Ripple frequency: {stats['ripple_frequency']/1e3:.1f} kHz")
print(f"Crest factor:     {stats['crest_factor']:.2f}")

# Check against specifications
if stats['ripple_pp_percent'] > 1.0:
    print("WARNING: Ripple exceeds 1% specification")

# Crest factor > 1.5 may indicate switching noise
if stats['crest_factor'] > 1.5:
    print("High crest factor - check for switching spikes")
```

#### Use Cases

- Power supply output quality verification
- Switching regulator noise analysis
- Battery charging ripple measurement
- EMI/EMC pre-compliance testing

## Complete Examples

### DC Power Supply Analysis

```python
import oscura as osc

# Load measurements
v_in = osc.load("supply_input_voltage.wfm")
i_in = osc.load("supply_input_current.wfm")
v_out = osc.load("supply_output_voltage.wfm")
i_out = osc.load("supply_output_current.wfm")

# Calculate efficiency
eta = osc.efficiency(v_in, i_in, v_out, i_out)
print(f"Efficiency: {eta*100:.1f}%")

# Analyze output ripple
ripple_stats = osc.ripple_statistics(v_out)
print(f"\nOutput Quality:")
print(f"  DC Level:   {ripple_stats['dc_level']:.3f} V")
print(f"  Ripple:     {ripple_stats['ripple_pp']*1e3:.1f} mV pk-pk ({ripple_stats['ripple_pp_percent']:.2f}%)")
print(f"  Frequency:  {ripple_stats['ripple_frequency']/1e3:.1f} kHz")

# Calculate power budget
p_in = osc.average_power(voltage=v_in, current=i_in)
p_out = osc.average_power(voltage=v_out, current=i_out)
e_in = osc.energy(voltage=v_in, current=i_in)

print(f"\nPower Budget:")
print(f"  Input power:  {p_in:.2f} W")
print(f"  Output power: {p_out:.2f} W")
print(f"  Losses:       {p_in - p_out:.2f} W")
print(f"  Input energy: {e_in*1e3:.1f} mJ")
```

### AC Motor Power Analysis

```python
import oscura as osc

# Load three-phase measurements
v_ac = osc.load("motor_voltage.wfm")
i_ac = osc.load("motor_current.wfm")

# Calculate AC power quantities
p = osc.average_power(voltage=v_ac, current=i_ac)
s = osc.apparent_power(v_ac, i_ac)
q = osc.reactive_power(v_ac, i_ac)
pf = osc.power_factor(v_ac, i_ac)

print("AC Motor Analysis:")
print(f"  Real Power:      {p:.2f} W")
print(f"  Apparent Power:  {s:.2f} VA")
print(f"  Reactive Power:  {q:.2f} VAR")
print(f"  Power Factor:    {pf:.3f}")

# Determine load characteristics
if q > 0:
    print(f"  Load Type:       Inductive (motor)")
    print(f"  Phase Angle:     {np.degrees(np.arcsin(q/s)):.1f}° lagging")
else:
    print(f"  Load Type:       Capacitive")

# Calculate required reactive power compensation
if pf < 0.95:
    target_pf = 0.95
    q_correction = p * (np.tan(np.arccos(pf)) - np.tan(np.arccos(target_pf)))
    capacitance = q_correction / (2 * np.pi * 60 * v_rms**2)
    print(f"\nPower Factor Correction:")
    print(f"  Required correction: {q_correction:.2f} VAR")
    print(f"  Capacitor size (60Hz): {capacitance*1e6:.1f} µF")
```

### Switching Converter Analysis

```python
import oscura as osc
import numpy as np

# Load converter measurements
v_in = osc.load("converter_vin.wfm")
i_in = osc.load("converter_iin.wfm")
v_out = osc.load("converter_vout.wfm")
i_out = osc.load("converter_iout.wfm")

# Efficiency calculation
eta = osc.efficiency(v_in, i_in, v_out, i_out)
print(f"Converter Efficiency: {eta*100:.1f}%")

# Output ripple analysis
ripple_stats = osc.ripple_statistics(v_out)
print(f"\nOutput Ripple:")
print(f"  Peak-to-peak: {ripple_stats['ripple_pp']*1e3:.2f} mV ({ripple_stats['ripple_pp_percent']:.3f}%)")
print(f"  RMS:          {ripple_stats['ripple_rms']*1e3:.2f} mV ({ripple_stats['ripple_rms_percent']:.3f}%)")
print(f"  Frequency:    {ripple_stats['ripple_frequency']/1e3:.1f} kHz")

# Input current ripple
i_ripple_pp, i_ripple_rms = osc.ripple(i_in)
print(f"\nInput Current Ripple:")
print(f"  Peak-to-peak: {i_ripple_pp*1e3:.2f} mA")
print(f"  RMS:          {i_ripple_rms*1e3:.2f} mA")

# Power statistics
power_stats = osc.power_statistics(voltage=v_in, current=i_in)
print(f"\nInput Power Statistics:")
print(f"  Average: {power_stats['average']:.2f} W")
print(f"  Peak:    {power_stats['peak']:.2f} W")
print(f"  Energy:  {power_stats['energy']*1e3:.1f} mJ")
```

### Battery Discharge Analysis

```python
import oscura as osc

# Load battery measurements
v_batt = osc.load("battery_voltage.wfm")
i_batt = osc.load("battery_current.wfm")

# Calculate total energy discharged
e_total = osc.energy(voltage=v_batt, current=i_batt)
print(f"Total energy discharged: {e_total:.2f} J ({e_total/3600:.2f} Wh)")

# Calculate average power draw
p_avg = osc.average_power(voltage=v_batt, current=i_batt)
print(f"Average power: {p_avg:.2f} W")

# Get power statistics
stats = osc.power_statistics(voltage=v_batt, current=i_batt)
print(f"\nPower Statistics:")
print(f"  Average:  {stats['average']:.2f} W")
print(f"  Peak:     {stats['peak']:.2f} W")
print(f"  Minimum:  {stats['min']:.2f} W")
print(f"  Std Dev:  {stats['std']:.2f} W")

# Estimate runtime at different loads
sample_rate = v_batt.metadata.sample_rate
duration = len(v_batt.data) / sample_rate
print(f"\nDischarge Profile:")
print(f"  Duration:     {duration:.1f} s")
print(f"  Energy rate:  {e_total/duration:.2f} W")

# Battery capacity estimation
nominal_voltage = 3.7  # V
capacity_mah = (e_total / 3600) / nominal_voltage * 1000
print(f"  Estimated capacity: {capacity_mah:.0f} mAh @ {nominal_voltage}V")
```

## Best Practices

### Sample Rate Considerations

```python
import oscura as osc

# Ensure adequate sample rate for power measurements
v = osc.load("voltage.wfm")
i = osc.load("current.wfm")

# Check sample rates
print(f"Voltage sample rate: {v.metadata.sample_rate/1e6:.1f} MS/s")
print(f"Current sample rate: {i.metadata.sample_rate/1e6:.1f} MS/s")

# For AC power analysis, need ~100 samples per cycle minimum
fundamental_freq = 60  # Hz
min_sample_rate = fundamental_freq * 100
if v.metadata.sample_rate < min_sample_rate:
    print(f"WARNING: Sample rate may be insufficient for accurate AC analysis")
    print(f"Recommended: {min_sample_rate/1e3:.1f} kS/s minimum")
```

### Handling Noisy Measurements

```python
import oscura as osc

# Load noisy measurements
v = osc.load("noisy_voltage.wfm")
i = osc.load("noisy_current.wfm")

# Apply filtering to reduce noise
v_filtered = osc.low_pass(v, cutoff=100e3)  # 100 kHz cutoff
i_filtered = osc.low_pass(i, cutoff=100e3)

# Calculate power with filtered signals
power_clean = osc.instantaneous_power(v_filtered, i_filtered)
power_noisy = osc.instantaneous_power(v, i)

# Compare results
p_clean = osc.average_power(power_clean)
p_noisy = osc.average_power(power_noisy)
print(f"Filtered: {p_clean:.2f} W")
print(f"Noisy:    {p_noisy:.2f} W")
print(f"Difference: {abs(p_clean - p_noisy):.2f} W")
```

### Time-Windowed Analysis

```python
import oscura as osc

# Analyze power during specific events
v = osc.load("voltage.wfm")
i = osc.load("current.wfm")

# Calculate power for entire trace
power = osc.instantaneous_power(v, i)

# Analyze different time windows
startup_energy = osc.energy(power, start_time=0, end_time=0.001)  # First 1ms
steady_energy = osc.energy(power, start_time=0.001, end_time=0.1)  # 1ms to 100ms

print(f"Startup energy: {startup_energy*1e6:.2f} µJ")
print(f"Steady-state energy: {steady_energy*1e3:.2f} mJ")

# Calculate average power in each phase
startup_power = startup_energy / 0.001
steady_power = steady_energy / (0.1 - 0.001)
print(f"Startup power: {startup_power:.2f} W")
print(f"Steady power: {steady_power:.2f} W")
```

### Ripple Frequency Analysis

```python
import oscura as osc
from oscura.analyzers.power.ripple import ripple_frequency, ripple_harmonics

# Analyze ripple frequency content
dc_out = osc.load("dc_output.wfm")

# Find dominant ripple frequency
f_ripple = ripple_frequency(dc_out)
print(f"Dominant ripple frequency: {f_ripple/1e3:.1f} kHz")

# Analyze harmonics
harmonics = ripple_harmonics(dc_out, fundamental_freq=f_ripple, n_harmonics=5)
print(f"\nRipple Harmonics:")
for h, amplitude in harmonics.items():
    print(f"  H{h} ({h*f_ripple/1e3:.1f} kHz): {amplitude*1e3:.2f} mV")
```

### Multi-Output Power Supply

```python
import oscura as osc
from oscura.analyzers.power.efficiency import multi_output_efficiency

# Load measurements for multi-output supply
v_in = osc.load("psu_input_voltage.wfm")
i_in = osc.load("psu_input_current.wfm")

# Load output measurements
v_5v = osc.load("output_5v_voltage.wfm")
i_5v = osc.load("output_5v_current.wfm")
v_12v = osc.load("output_12v_voltage.wfm")
i_12v = osc.load("output_12v_current.wfm")
v_3v3 = osc.load("output_3v3_voltage.wfm")
i_3v3 = osc.load("output_3v3_current.wfm")

# Calculate multi-output efficiency
outputs = [(v_5v, i_5v), (v_12v, i_12v), (v_3v3, i_3v3)]
result = multi_output_efficiency(v_in, i_in, outputs)

print(f"Total Efficiency: {result['total_efficiency']*100:.1f}%")
print(f"Total Output Power: {result['total_output_power']:.2f} W")
print(f"Input Power: {result['input_power']:.2f} W")
print(f"Losses: {result['losses']:.2f} W")

print(f"\nPer-Output Analysis:")
for i in range(len(outputs)):
    print(f"  Output {i+1}: {result[f'output_{i+1}_power']:.2f} W "
          f"({result[f'output_{i+1}_efficiency']*100:.1f}%)")
```

## Error Handling

```python
import oscura as osc

try:
    # Attempt power calculation
    power = osc.instantaneous_power(v, i, interpolate_if_needed=False)
except osc.AnalysisError as e:
    print(f"Analysis error: {e}")
    # Enable interpolation or check traces
    power = osc.instantaneous_power(v, i, interpolate_if_needed=True)

# Check for valid power measurements
stats = osc.power_statistics(voltage=v, current=i)
if stats['average'] < 0:
    print("WARNING: Negative average power - check probe polarity")

# Validate ripple measurements
r_pp, r_rms = osc.ripple(dc_output)
if r_rms > r_pp / 2:
    print("WARNING: Unusual ripple - may not be sinusoidal")
```

## API Reference Summary

### Basic Power Functions

- `osc.instantaneous_power(voltage, current, *, interpolate_if_needed=True)` - Calculate P(t) = V(t) × I(t)
- `osc.average_power(power=None, *, voltage=None, current=None)` - Calculate mean power
- `osc.energy(power=None, *, voltage=None, current=None, start_time=None, end_time=None)` - Calculate total energy
- `osc.power_statistics(power=None, *, voltage=None, current=None)` - Comprehensive power stats

### AC Power Functions

- `osc.apparent_power(voltage, current)` - Calculate S = V_rms × I_rms (VA)
- `osc.reactive_power(voltage, current, *, frequency=None)` - Calculate Q = S × sin(φ) (VAR)
- `osc.power_factor(voltage, current)` - Calculate PF = P / S

### Efficiency Functions

- `osc.efficiency(v_in, i_in, v_out, i_out)` - Calculate η = P_out / P_in

### Ripple Functions

- `osc.ripple(trace, *, dc_coupling=False)` - Measure AC ripple (peak-peak and RMS)
- `osc.ripple_statistics(trace)` - Comprehensive ripple analysis

## See Also

- [Analysis API](analysis.md) - General waveform analysis and filtering
- [Export API](export.md) - Exporting power analysis results
- [Visualization API](visualization.md) - Plotting power waveforms
