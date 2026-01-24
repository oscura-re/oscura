# Oscura Workflow Helpers API Documentation

> **Version**: 0.5.1 | **Last Updated**: 2026-01-19

Complete guide to high-level workflow functions that combine multiple Oscura features for common analysis tasks.

## Overview

Oscura provides pre-built workflow functions that combine multiple analysis steps into single function calls. These workflows automate common measurement tasks and provide comprehensive results:

- **Buffer Characterization** - Complete digital buffer analysis with automatic logic family detection
- **Protocol Debugging** - Auto-detect and decode protocols with error context
- **EMC Compliance Testing** - Spectral compliance testing against regulatory standards
- **Power Analysis** - Comprehensive power consumption and efficiency analysis
- **Signal Integrity Audit** - Complete signal integrity testing with eye diagrams and jitter

All workflow functions are accessible via top-level API: `osc.characterize_buffer()`, `osc.debug_protocol()`, etc.

## Quick Start

```python
import oscura as osc

# Buffer characterization workflow
trace = osc.load("74hc04_output.wfm")
result = osc.characterize_buffer(trace)
print(f"Logic family: {result['logic_family']}, Status: {result['status']}")

# Protocol debugging workflow
serial = osc.load("uart_capture.wfm")
result = osc.debug_protocol(serial)
print(f"Protocol: {result['protocol']}, Errors: {len(result['errors'])}")

# EMC compliance workflow
emissions = osc.load("radiated_emissions.wfm")
result = osc.emc_compliance_test(emissions, standard='FCC_Part15_ClassB')
print(f"Compliance: {result['status']}")

# Power analysis workflow
voltage = osc.load("vdd.wfm")
current = osc.load("idd.wfm")
result = osc.power_analysis(voltage, current)
print(f"Average power: {result['average_power']*1e3:.2f} mW")

# Signal integrity workflow
data = osc.load("high_speed_data.wfm")
result = osc.signal_integrity_audit(data, bit_rate=1e9)
print(f"Eye height: {result['eye_height']*1e3:.1f} mV")
```

## Buffer Characterization Workflow

### characterize_buffer()

Complete digital buffer characterization in a single function call.

#### Purpose

Performs comprehensive analysis of digital buffer output signals including:

- Automatic logic family detection (TTL, CMOS, LVTTL, etc.)
- Rise/fall time measurements
- Propagation delay measurement (if reference provided)
- Overshoot/undershoot analysis
- Noise margin calculation
- Pass/fail determination against logic family specifications

Use this workflow when testing digital ICs, evaluating buffer performance, or verifying logic level compliance.

#### Function Signature

```python
osc.characterize_buffer(
    trace,                          # Output signal to characterize
    *,
    reference_trace=None,           # Optional reference (input) signal
    logic_family=None,              # Override logic family detection
    thresholds=None,                # Custom pass/fail thresholds
    report=None                     # Optional HTML report path
) -> dict[str, Any]
```

#### Parameters

- **trace** (`WaveformTrace`) - Output signal to characterize. Should contain at least one complete digital transition.
- **reference_trace** (`WaveformTrace`, optional) - Optional reference (input) signal for propagation delay measurement. Must have same sample rate as trace.
- **logic_family** (`str`, optional) - Override automatic logic family detection. Options: `'TTL'`, `'CMOS_5V'`, `'CMOS_3V3'`, `'LVTTL'`, `'LVCMOS'`. If `None`, auto-detected from signal levels.
- **thresholds** (`dict[str, float]`, optional) - Custom pass/fail thresholds. Example: `{'rise_time': 10e-9, 'fall_time': 10e-9, 'overshoot_percent': 20.0}`.
- **report** (`str`, optional) - Path to save HTML report with results.

#### Returns

`dict[str, Any]` - Dictionary containing:

- **logic_family** (`str`) - Detected or specified logic family
- **confidence** (`float`) - Detection confidence (0-1)
- **rise_time** (`float`) - 10%-90% rise time in seconds
- **fall_time** (`float`) - 10%-90% fall time in seconds
- **propagation_delay** (`float` or `None`) - Delay from reference in seconds (if reference provided)
- **overshoot** (`float`) - Peak overshoot voltage
- **overshoot_percent** (`float`) - Overshoot as percentage of swing
- **undershoot** (`float`) - Peak undershoot voltage
- **undershoot_percent** (`float`) - Undershoot as percentage of swing
- **noise_margin_high** (`float`) - High-level noise margin in volts
- **noise_margin_low** (`float`) - Low-level noise margin in volts
- **voh** (`float`) - Measured VOH in volts
- **vol** (`float`) - Measured VOL in volts
- **status** (`str`) - `'PASS'` or `'FAIL'` based on logic family specs
- **reference_comparison** (`dict` or `None`) - Reference timing comparison if provided

#### Examples

##### Basic Buffer Characterization

```python
import oscura as osc

# Load buffer output
trace = osc.load("74hc04_output.wfm")

# Characterize with auto-detection
result = osc.characterize_buffer(trace)

print(f"Logic Family: {result['logic_family']} (confidence: {result['confidence']:.1%})")
print(f"Rise Time:    {result['rise_time']*1e9:.2f} ns")
print(f"Fall Time:    {result['fall_time']*1e9:.2f} ns")
print(f"Overshoot:    {result['overshoot_percent']:.1f}%")
print(f"Undershoot:   {result['undershoot_percent']:.1f}%")
print(f"Status:       {result['status']}")
```

##### Characterization with Reference Signal

```python
import oscura as osc

# Load input and output
input_sig = osc.load("buffer_input.wfm")
output_sig = osc.load("buffer_output.wfm")

# Characterize with propagation delay
result = osc.characterize_buffer(
    output_sig,
    reference_trace=input_sig
)

print(f"Propagation Delay: {result['propagation_delay']*1e9:.2f} ns")
print(f"Rise Time:         {result['rise_time']*1e9:.2f} ns")
print(f"Fall Time:         {result['fall_time']*1e9:.2f} ns")
```

##### Custom Thresholds and Report Generation

```python
import oscura as osc

# Load signal
trace = osc.load("custom_logic.wfm")

# Characterize with custom thresholds
result = osc.characterize_buffer(
    trace,
    logic_family='CMOS_3V3',
    thresholds={
        'rise_time': 5e-9,        # 5 ns max
        'fall_time': 5e-9,        # 5 ns max
        'overshoot_percent': 15.0  # 15% max
    },
    report='buffer_report.html'
)

if result['status'] == 'FAIL':
    print("Failed requirements:")
    if result['rise_time'] > 5e-9:
        print(f"  Rise time: {result['rise_time']*1e9:.2f} ns (max 5.00 ns)")
    if result['fall_time'] > 5e-9:
        print(f"  Fall time: {result['fall_time']*1e9:.2f} ns (max 5.00 ns)")
    if result['overshoot_percent'] > 15.0:
        print(f"  Overshoot: {result['overshoot_percent']:.1f}% (max 15.0%)")
```

##### Batch Characterization

```python
import oscura as osc

# Test multiple buffer outputs
buffers = ['74hc04', '74hc14', '74hc244']
results = {}

for buffer in buffers:
    trace = osc.load(f"{buffer}_output.wfm")
    results[buffer] = osc.characterize_buffer(trace)

# Compare results
print("Buffer Comparison:")
print(f"{'Buffer':<10} {'Family':<10} {'Rise (ns)':<10} {'Status'}")
for name, result in results.items():
    print(f"{name:<10} {result['logic_family']:<10} "
          f"{result['rise_time']*1e9:<10.2f} {result['status']}")
```

#### Analysis Steps Performed

The `characterize_buffer()` workflow performs these analysis steps:

1. **Logic Family Detection** (if not specified)
   - Measures VOH and VOL from signal levels
   - Compares against standard logic family thresholds
   - Returns confidence score based on match quality

2. **Rise/Fall Time Measurement**
   - Detects rising and falling edges
   - Measures 10%-90% transition times
   - Uses IEEE 181-2011 standard definitions

3. **Overshoot/Undershoot Analysis**
   - Identifies peaks beyond nominal levels
   - Calculates percentage relative to signal swing
   - Checks for ringing and oscillations

4. **Noise Margin Calculation**
   - Compares VOH/VOL to VIH/VIL thresholds
   - Calculates high and low noise margins
   - Based on logic family specifications

5. **Propagation Delay** (if reference provided)
   - Cross-correlates input and output signals
   - Measures delay at 50% threshold
   - Reports timing drift over measurement

6. **Pass/Fail Determination**
   - Compares measurements against thresholds
   - Uses logic family specs or custom thresholds
   - Generates comprehensive status report

#### Use Cases

- **Buffer IC Testing** - Verify 74HC, 74AC, 74LVC buffer performance
- **Logic Level Translation** - Test level shifters and translators
- **Driver Characterization** - Analyze line drivers and transceivers
- **Signal Quality Assessment** - Evaluate output signal quality
- **Production Testing** - Automated pass/fail testing

#### References

- IEEE 181-2011: Standard for Transitional Waveform Definitions
- JEDEC Standard No. 65B: High-Speed Interface Timing

## Protocol Debugging Workflow

### debug_protocol()

Auto-detect and decode protocol with error context and analysis.

#### Purpose

Automatically detects protocol type and decodes packets with comprehensive error reporting:

- Auto-detection of protocol type (UART, SPI, I2C, CAN)
- Automatic baud rate / clock rate detection
- Packet decoding with error detection
- Context samples around errors for debugging
- Statistical analysis of error rates

Use this workflow when debugging communication issues, analyzing protocol errors, or reverse-engineering unknown protocols.

#### Function Signature

```python
osc.debug_protocol(
    trace,                          # Signal to decode
    *,
    protocol=None,                  # Protocol type override
    context_samples=100,            # Samples around errors
    error_types=None,               # Filter error types
    decode_all=False                # Decode all packets or only errors
) -> dict[str, Any]
```

#### Parameters

- **trace** (`WaveformTrace` or `DigitalTrace`) - Signal to decode. Can be analog (auto-thresholded) or digital.
- **protocol** (`str`, optional) - Protocol type override: `'UART'`, `'SPI'`, `'I2C'`, `'CAN'`, or `'auto'`. If `None` or `'auto'`, protocol is auto-detected.
- **context_samples** (`int`, optional) - Number of samples to include before/after errors for debugging context. Default: 100.
- **error_types** (`list[str]`, optional) - List of error types to detect. If `None`, detects all errors (framing, parity, stop bit, NAK, CRC, etc.).
- **decode_all** (`bool`, optional) - If `True`, decode all packets. If `False` (default), focus on packets with errors.

#### Returns

`dict[str, Any]` - Dictionary containing:

- **protocol** (`str`) - Detected or specified protocol type
- **baud_rate** (`float` or `None`) - Detected baud/clock rate in Hz (if applicable)
- **packets** (`list[ProtocolPacket]`) - List of decoded packet objects
- **errors** (`list[dict]`) - List of error dictionaries with context:
  - **type** - Error type string
  - **timestamp** - Time of error in seconds
  - **packet_index** - Index of packet with error
  - **address** - Address field (if applicable)
  - **data** - Data field
  - **context** - Description of context window
  - **context_trace** - Sub-trace with context samples
- **config** (`dict`) - Protocol configuration used for decoding
- **statistics** (`dict`) - Decoding statistics:
  - **total_packets** - Total packets decoded
  - **error_count** - Number of errors found
  - **error_rate** - Ratio of errors to packets
  - **confidence** - Protocol detection confidence (0-1)

#### Examples

##### Auto-Detect and Debug UART

```python
import oscura as osc

# Load UART capture
trace = osc.load("uart_capture.wfm")

# Auto-detect and decode
result = osc.debug_protocol(trace)

print(f"Protocol:   {result['protocol']}")
print(f"Baud Rate:  {result['baud_rate']} baud")
print(f"Packets:    {result['statistics']['total_packets']}")
print(f"Errors:     {result['statistics']['error_count']}")
print(f"Error Rate: {result['statistics']['error_rate']*100:.1f}%")

# List errors with context
for error in result['errors']:
    print(f"\nError at {error['timestamp']*1e6:.2f} µs:")
    print(f"  Type: {error['type']}")
    print(f"  Data: {error['data']:#x}")
    print(f"  {error['context']}")
```

##### Force Protocol Type and Configuration

```python
import oscura as osc

# Load I2C capture
trace = osc.load("i2c_capture.wfm")

# Force I2C decoding
result = osc.debug_protocol(
    trace,
    protocol='I2C',
    context_samples=200  # More context for debugging
)

print(f"I2C Decoding Results:")
for i, packet in enumerate(result['packets']):
    addr = packet.annotations.get('address', 0) if packet.annotations else 0
    rw = 'Read' if (addr & 1) else 'Write'
    print(f"Packet {i}: Addr 0x{addr>>1:02x} {rw}, Data: {packet.data.hex()}")
    if packet.errors:
        print(f"  Errors: {', '.join(packet.errors)}")
```

##### Filter Specific Error Types

```python
import oscura as osc

# Load SPI capture
trace = osc.load("spi_capture.wfm")

# Debug only framing and parity errors
result = osc.debug_protocol(
    trace,
    protocol='SPI',
    error_types=['framing', 'parity'],
    decode_all=True  # Get all packets, not just errors
)

# Analyze error distribution
framing_errors = [e for e in result['errors'] if 'framing' in e['type'].lower()]
parity_errors = [e for e in result['errors'] if 'parity' in e['type'].lower()]

print(f"Framing Errors: {len(framing_errors)}")
print(f"Parity Errors:  {len(parity_errors)}")

# Plot packets with errors
import matplotlib.pyplot as plt
error_times = [e['timestamp'] for e in result['errors']]
plt.scatter(error_times, [1]*len(error_times), marker='x', color='red')
plt.xlabel('Time (s)')
plt.title('Error Distribution')
plt.show()
```

##### Extract Context for Deep Debugging

```python
import oscura as osc

# Load problematic capture
trace = osc.load("uart_issues.wfm")

# Debug with large context window
result = osc.debug_protocol(
    trace,
    context_samples=500,
    decode_all=False
)

# Export error contexts for detailed analysis
for i, error in enumerate(result['errors']):
    context_trace = error['context_trace']
    if context_trace is not None:
        # Save context around each error
        osc.save(context_trace, f"error_{i}_context.wfm")

        # Analyze context
        print(f"\nError {i} at {error['timestamp']*1e6:.2f} µs:")
        print(f"  Type: {error['type']}")

        # Check for noise or glitches in context
        if isinstance(context_trace, osc.WaveformTrace):
            noise = osc.std(context_trace)
            print(f"  Noise level: {noise*1e3:.2f} mV")
```

##### Batch Protocol Analysis

```python
import oscura as osc

# Analyze multiple captures
captures = [
    'capture_session1.wfm',
    'capture_session2.wfm',
    'capture_session3.wfm'
]

summary = []
for capture in captures:
    trace = osc.load(capture)
    result = osc.debug_protocol(trace)

    summary.append({
        'file': capture,
        'protocol': result['protocol'],
        'baud_rate': result['baud_rate'],
        'packets': result['statistics']['total_packets'],
        'errors': result['statistics']['error_count'],
        'error_rate': result['statistics']['error_rate']
    })

# Print comparison
print("Protocol Analysis Summary:")
print(f"{'File':<25} {'Protocol':<8} {'Packets':<10} {'Errors':<10} {'Rate'}")
for s in summary:
    print(f"{s['file']:<25} {s['protocol']:<8} {s['packets']:<10} "
          f"{s['errors']:<10} {s['error_rate']*100:>5.1f}%")
```

#### Analysis Steps Performed

The `debug_protocol()` workflow performs these analysis steps:

1. **Protocol Detection** (if not specified)
   - Analyzes edge patterns and timing
   - Detects characteristic protocol signatures
   - Returns protocol type and confidence score
   - Estimates baud/clock rate

2. **Signal Conditioning**
   - Converts analog to digital if needed
   - Auto-thresholds waveform at mid-level
   - Filters glitches and noise

3. **Protocol Decoding**
   - Applies protocol-specific decoder
   - Extracts packets with timestamps
   - Decodes data fields and annotations
   - Detects protocol errors

4. **Error Detection**
   - Identifies framing errors (UART)
   - Detects parity errors (UART)
   - Finds missing ACK/NAK (I2C)
   - Checks CRC/checksum (CAN, others)
   - Detects timing violations

5. **Context Extraction**
   - Captures samples around each error
   - Provides before/after context window
   - Enables detailed error investigation

6. **Statistical Analysis**
   - Calculates error rates
   - Counts total packets
   - Measures protocol quality metrics

#### Use Cases

- **Protocol Debugging** - Identify communication errors
- **Bus Analysis** - Debug I2C, SPI, UART, CAN buses
- **Error Characterization** - Analyze error patterns and rates
- **Protocol Reverse Engineering** - Discover unknown protocols
- **Quality Testing** - Validate communication reliability

#### Supported Protocols

- **UART** - RS-232, TTL serial, various baud rates
- **SPI** - All CPOL/CPHA modes, configurable word sizes
- **I2C** - 7-bit and 10-bit addressing, all speeds
- **CAN** - Standard and extended frames

#### References

- UART: TIA-232-F (Serial communication)
- I2C: NXP UM10204 (I2C-bus specification)
- SPI: Motorola SPI Block Guide
- CAN: ISO 11898
- sigrok Protocol Decoder API

## EMC Compliance Testing Workflow

### emc_compliance_test()

EMC/EMI compliance testing against regulatory limits.

#### Purpose

Performs spectral compliance testing for electromagnetic compatibility:

- Computes frequency spectrum (FFT)
- Loads regulatory limit masks
- Overlays limit lines on spectrum
- Identifies frequency violations
- Generates compliance reports

Use this workflow when testing products for EMC certification, debugging emissions issues, or performing pre-compliance testing.

#### Function Signature

```python
osc.emc_compliance_test(
    trace,                          # Signal to test
    *,
    standard='FCC_Part15_ClassB',   # Regulatory standard
    frequency_range=None,           # Optional freq range
    detector='peak',                # Detector type
    report=None                     # Optional HTML report
) -> dict[str, Any]
```

#### Parameters

- **trace** (`WaveformTrace`) - Signal to test for emissions. Typically from near-field probe or antenna.
- **standard** (`str`, optional) - Regulatory standard to test against:
  - `'FCC_Part15_ClassA'` - FCC Part 15 Class A (commercial)
  - `'FCC_Part15_ClassB'` - FCC Part 15 Class B (residential, default)
  - `'CE_CISPR22_ClassA'` - CISPR 22 Class A (commercial)
  - `'CE_CISPR22_ClassB'` - CISPR 22 Class B (residential)
  - `'CE_CISPR32_ClassA'` - CISPR 32 Class A (commercial)
  - `'CE_CISPR32_ClassB'` - CISPR 32 Class B (residential)
  - `'MIL_STD_461G_CE102'` - Military conducted emissions
  - `'MIL_STD_461G_RE102'` - Military radiated emissions
- **frequency_range** (`tuple[float, float]`, optional) - Frequency range to test `(f_min, f_max)` in Hz. If `None`, tests full spectrum.
- **detector** (`str`, optional) - Detector type: `'peak'` (default), `'quasi-peak'`, or `'average'`.
- **report** (`str`, optional) - Path to save HTML compliance report.

#### Returns

`dict[str, Any]` - Dictionary containing:

- **status** (`str`) - `'PASS'` or `'FAIL'`
- **standard** (`str`) - Standard tested against
- **violations** (`list[dict]`) - List of frequency violations:
  - **frequency** - Violation frequency in Hz
  - **measured_dbuv** - Measured level in dBµV
  - **limit_dbuv** - Limit level in dBµV
  - **excess_db** - Excess above limit in dB (positive value)
- **margin_to_limit** (`float`) - Minimum margin in dB (negative if failing)
- **worst_frequency** (`float`) - Frequency with worst margin in Hz
- **worst_margin** (`float`) - Worst margin value in dB
- **spectrum_freq** (`ndarray`) - Frequency array for spectrum in Hz
- **spectrum_mag** (`ndarray`) - Magnitude array in dBµV
- **limit_freq** (`ndarray`) - Frequency array for limit mask in Hz
- **limit_mag** (`ndarray`) - Magnitude array for limit mask in dBµV
- **detector** (`str`) - Detector type used

#### Examples

##### Basic FCC Part 15 Compliance Test

```python
import oscura as osc

# Load radiated emissions measurement
trace = osc.load("radiated_emissions.wfm")

# Test FCC Part 15 Class B compliance
result = osc.emc_compliance_test(
    trace,
    standard='FCC_Part15_ClassB'
)

print(f"Compliance Status: {result['status']}")
print(f"Margin to Limit:   {result['margin_to_limit']:.1f} dB")
print(f"Worst Frequency:   {result['worst_frequency']/1e6:.2f} MHz")

if result['status'] == 'FAIL':
    print(f"\nViolations found: {len(result['violations'])}")
    for v in result['violations']:
        print(f"  {v['frequency']/1e6:.2f} MHz: "
              f"{v['measured_dbuv']:.1f} dBµV "
              f"(limit: {v['limit_dbuv']:.1f} dBµV, "
              f"excess: {v['excess_db']:.1f} dB)")
```

##### CISPR 32 Compliance with Report

```python
import oscura as osc

# Load emissions
trace = osc.load("product_emissions.wfm")

# Test CISPR 32 Class B
result = osc.emc_compliance_test(
    trace,
    standard='CE_CISPR32_ClassB',
    report='cispr32_compliance_report.html'
)

print(f"CISPR 32 Class B Compliance: {result['status']}")

if result['status'] == 'PASS':
    print(f"Margin: {result['margin_to_limit']:.1f} dB")
    print("Product complies with CISPR 32 Class B limits")
else:
    print(f"Failed at {result['worst_frequency']/1e6:.2f} MHz")
    print(f"Margin: {result['worst_margin']:.1f} dB")
    print("\nRemediation required:")
    for v in result['violations'][:5]:  # Show first 5
        print(f"  {v['frequency']/1e6:.2f} MHz needs "
              f"{v['excess_db']:.1f} dB reduction")
```

##### Frequency Range Testing

```python
import oscura as osc

# Load emissions
trace = osc.load("emissions.wfm")

# Test only specific frequency range (30 MHz - 1 GHz)
result = osc.emc_compliance_test(
    trace,
    standard='FCC_Part15_ClassB',
    frequency_range=(30e6, 1e9)
)

print(f"30 MHz - 1 GHz Range:")
print(f"  Status: {result['status']}")
print(f"  Margin: {result['margin_to_limit']:.1f} dB")
```

##### Plot Spectrum with Limit Lines

```python
import oscura as osc
import matplotlib.pyplot as plt

# Load and test
trace = osc.load("emissions.wfm")
result = osc.emc_compliance_test(trace, standard='FCC_Part15_ClassB')

# Plot spectrum and limits
plt.figure(figsize=(12, 6))
plt.plot(result['spectrum_freq']/1e6, result['spectrum_mag'],
         label='Measured', linewidth=1)
plt.plot(result['limit_freq']/1e6, result['limit_mag'],
         'r--', label='FCC Part 15 Class B Limit', linewidth=2)

# Mark violations
if result['violations']:
    violation_freqs = [v['frequency']/1e6 for v in result['violations']]
    violation_mags = [v['measured_dbuv'] for v in result['violations']]
    plt.scatter(violation_freqs, violation_mags,
                color='red', s=100, marker='x', label='Violations')

plt.xlabel('Frequency (MHz)')
plt.ylabel('Amplitude (dBµV)')
plt.title(f"EMC Compliance Test - Status: {result['status']}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.show()
```

##### Military Standard Testing

```python
import oscura as osc

# Load conducted emissions measurement
trace = osc.load("power_line_emissions.wfm")

# Test MIL-STD-461G CE102
result = osc.emc_compliance_test(
    trace,
    standard='MIL_STD_461G_CE102',
    detector='average'  # Military standards often use average detector
)

print(f"MIL-STD-461G CE102 Compliance: {result['status']}")
print(f"Margin: {result['margin_to_limit']:.1f} dB")

# Check low-frequency violations (common problem area)
low_freq_violations = [v for v in result['violations']
                       if v['frequency'] < 1e6]
if low_freq_violations:
    print(f"\nLow-frequency violations (<1 MHz): {len(low_freq_violations)}")
    print("Consider input filtering or shielding")
```

##### Comparative Testing

```python
import oscura as osc

# Test against multiple standards
trace = osc.load("emissions.wfm")

standards = [
    'FCC_Part15_ClassA',
    'FCC_Part15_ClassB',
    'CE_CISPR32_ClassA',
    'CE_CISPR32_ClassB'
]

print("Multi-Standard Compliance Test:")
for std in standards:
    result = osc.emc_compliance_test(trace, standard=std)
    status_icon = '✓' if result['status'] == 'PASS' else '✗'
    print(f"{status_icon} {std:<25} {result['status']:<6} "
          f"Margin: {result['margin_to_limit']:>6.1f} dB")
```

#### Analysis Steps Performed

The `emc_compliance_test()` workflow performs these analysis steps:

1. **Spectrum Computation**
   - Calculates FFT of input signal
   - Converts to dBµV units
   - Applies appropriate windowing
   - Handles DC component

2. **Limit Mask Loading**
   - Loads regulatory limit data for standard
   - Interpolates limit to spectrum frequencies
   - Handles frequency-dependent limits

3. **Compliance Checking**
   - Compares spectrum to limits at all frequencies
   - Identifies violations (spectrum exceeds limit)
   - Calculates margins at all points

4. **Violation Analysis**
   - Lists all violation frequencies
   - Calculates excess above limit
   - Identifies worst-case frequency

5. **Statistical Analysis**
   - Calculates overall margin to limit
   - Determines pass/fail status
   - Counts violation points

6. **Report Generation** (if requested)
   - Creates HTML compliance report
   - Includes spectrum plot with limits
   - Lists all violations with details

#### Use Cases

- **Pre-Compliance Testing** - Before formal EMC testing
- **Debugging Emissions** - Identify problematic frequencies
- **Design Verification** - Verify EMC design techniques
- **Continuous Testing** - Production line EMC checks
- **Multiple Standard Testing** - Test global product compliance

#### Regulatory Standards Reference

| Standard            | Application              | Frequency Range | Measurement        |
| ------------------- | ------------------------ | --------------- | ------------------ |
| FCC Part 15 Class A | Commercial/Industrial    | 150 kHz - 1 GHz | Radiated           |
| FCC Part 15 Class B | Residential              | 150 kHz - 1 GHz | Radiated           |
| CISPR 22/32 Class A | Commercial IT Equipment  | 150 kHz - 1 GHz | Conducted/Radiated |
| CISPR 22/32 Class B | Residential IT Equipment | 150 kHz - 1 GHz | Conducted/Radiated |
| MIL-STD-461G CE102  | Military Conducted       | 10 kHz - 50 MHz | Conducted          |
| MIL-STD-461G RE102  | Military Radiated        | 2 MHz - 18 GHz  | Radiated           |

#### References

- FCC Part 15 Subpart B: Unintentional Radiators
- CISPR 22/32: Information Technology Equipment EMC
- MIL-STD-461G: Requirements for the Control of Electromagnetic Interference Characteristics of Subsystems and Equipment

## Power Analysis Workflow

### power_analysis()

Comprehensive power consumption analysis workflow.

#### Purpose

Performs complete power consumption analysis from voltage and current measurements:

- Instantaneous power calculation P(t) = V(t) × I(t)
- Average, RMS, and peak power statistics
- Total energy consumption
- Power conversion efficiency (if input provided)
- Power loss calculation
- Optional HTML report generation

Use this workflow when characterizing power supplies, measuring device power consumption, or analyzing power converter efficiency.

#### Function Signature

```python
osc.power_analysis(
    voltage,                        # Output voltage trace
    current,                        # Output current trace
    *,
    input_voltage=None,             # Optional input voltage
    input_current=None,             # Optional input current
    report=None                     # Optional HTML report
) -> dict[str, Any]
```

#### Parameters

- **voltage** (`WaveformTrace`) - Output voltage trace in volts.
- **current** (`WaveformTrace`) - Output current trace in amperes.
- **input_voltage** (`WaveformTrace`, optional) - Input voltage for efficiency calculation.
- **input_current** (`WaveformTrace`, optional) - Input current for efficiency calculation.
- **report** (`str`, optional) - Path to save HTML report.

Must provide both `input_voltage` and `input_current` together for efficiency calculation.

#### Returns

`dict[str, Any]` - Dictionary containing:

- **power_trace** (`WaveformTrace`) - Instantaneous power waveform P(t)
- **average_power** (`float`) - Mean power in watts
- **output_power_avg** (`float`) - Average output power (same as average_power)
- **output_power_rms** (`float`) - RMS output power in watts
- **peak_power** (`float`) - Maximum power in watts
- **min_power** (`float`) - Minimum power in watts
- **energy** (`float`) - Total energy in joules
- **duration** (`float`) - Measurement duration in seconds
- **efficiency** (`float`, optional) - Efficiency percentage (if input provided)
- **power_loss** (`float`, optional) - Power loss in watts (if input provided)
- **input_power_avg** (`float`, optional) - Average input power (if input provided)

#### Examples

##### Basic Power Measurement

```python
import oscura as osc

# Load voltage and current
v_out = osc.load("output_voltage.wfm")
i_out = osc.load("output_current.wfm")

# Analyze power
result = osc.power_analysis(v_out, i_out)

print(f"Average Power: {result['average_power']*1e3:.2f} mW")
print(f"RMS Power:     {result['output_power_rms']*1e3:.2f} mW")
print(f"Peak Power:    {result['peak_power']*1e3:.2f} mW")
print(f"Energy:        {result['energy']*1e6:.2f} µJ")
print(f"Duration:      {result['duration']*1e3:.2f} ms")
```

##### Power Converter Efficiency Analysis

```python
import oscura as osc

# Load input and output measurements
v_in = osc.load("input_voltage.wfm")
i_in = osc.load("input_current.wfm")
v_out = osc.load("output_voltage.wfm")
i_out = osc.load("output_current.wfm")

# Analyze with efficiency calculation
result = osc.power_analysis(
    v_out, i_out,
    input_voltage=v_in,
    input_current=i_in,
    report='power_analysis.html'
)

print(f"Input Power:    {result['input_power_avg']*1e3:.2f} mW")
print(f"Output Power:   {result['output_power_avg']*1e3:.2f} mW")
print(f"Power Loss:     {result['power_loss']*1e3:.2f} mW")
print(f"Efficiency:     {result['efficiency']:.1f}%")
```

##### Battery Power Profile

```python
import oscura as osc
import matplotlib.pyplot as plt

# Load battery discharge
v_batt = osc.load("battery_voltage.wfm")
i_batt = osc.load("battery_current.wfm")

# Analyze power consumption
result = osc.power_analysis(v_batt, i_batt)

# Plot power profile
power_trace = result['power_trace']
sample_rate = power_trace.metadata.sample_rate
time = np.arange(len(power_trace.data)) / sample_rate

plt.figure(figsize=(12, 6))
plt.plot(time*1e3, power_trace.data*1e3)
plt.axhline(result['average_power']*1e3, color='r',
            linestyle='--', label=f"Average: {result['average_power']*1e3:.2f} mW")
plt.xlabel('Time (ms)')
plt.ylabel('Power (mW)')
plt.title('Battery Power Consumption Profile')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Calculate battery life
battery_capacity_mah = 2000  # mAh
avg_current_ma = result['average_power'] / np.mean(v_batt.data) * 1000
runtime_hours = battery_capacity_mah / avg_current_ma
print(f"\nEstimated Runtime: {runtime_hours:.1f} hours")
```

##### Load Step Response

```python
import oscura as osc

# Load power supply measurements during load step
v = osc.load("supply_voltage_load_step.wfm")
i = osc.load("supply_current_load_step.wfm")

# Analyze full trace
result = osc.power_analysis(v, i)

# Analyze time windows
sample_rate = v.metadata.sample_rate
step_time = 0.001  # Load step at 1 ms

# Before load step
v_before = osc.WaveformTrace(
    data=v.data[:int(step_time * sample_rate)],
    metadata=v.metadata
)
i_before = osc.WaveformTrace(
    data=i.data[:int(step_time * sample_rate)],
    metadata=i.metadata
)
result_before = osc.power_analysis(v_before, i_before)

# After load step
v_after = osc.WaveformTrace(
    data=v.data[int(step_time * sample_rate):],
    metadata=v.metadata
)
i_after = osc.WaveformTrace(
    data=i.data[int(step_time * sample_rate):],
    metadata=i.metadata
)
result_after = osc.power_analysis(v_after, i_after)

print("Load Step Analysis:")
print(f"Before: {result_before['average_power']*1e3:.2f} mW")
print(f"After:  {result_after['average_power']*1e3:.2f} mW")
print(f"Change: {(result_after['average_power'] - result_before['average_power'])*1e3:.2f} mW")
```

##### Multi-Channel Power Analysis

```python
import oscura as osc

# Analyze multiple power rails
rails = {
    '3.3V': ('vdd_3v3.wfm', 'idd_3v3.wfm'),
    '5.0V': ('vdd_5v.wfm', 'idd_5v.wfm'),
    '12V':  ('vdd_12v.wfm', 'idd_12v.wfm')
}

total_power = 0
results = {}

print("Power Rail Analysis:")
for rail_name, (v_file, i_file) in rails.items():
    v = osc.load(v_file)
    i = osc.load(i_file)
    result = osc.power_analysis(v, i)
    results[rail_name] = result

    power_mw = result['average_power'] * 1e3
    total_power += result['average_power']

    print(f"{rail_name:>6}: {power_mw:>8.2f} mW "
          f"(Peak: {result['peak_power']*1e3:>8.2f} mW)")

print(f"\nTotal: {total_power*1e3:>8.2f} mW")
```

#### Analysis Steps Performed

The `power_analysis()` workflow performs these analysis steps:

1. **Trace Validation**
   - Checks sample rate compatibility
   - Validates trace lengths
   - Ensures time alignment

2. **Instantaneous Power Calculation**
   - Calculates P(t) = V(t) × I(t) sample-by-sample
   - Creates power waveform trace
   - Preserves timing information

3. **Power Statistics**
   - Average power (mean of P(t))
   - RMS power (root-mean-square)
   - Peak power (maximum)
   - Minimum power

4. **Energy Calculation**
   - Integrates power over time
   - Total energy in joules
   - Measurement duration

5. **Efficiency Analysis** (if input provided)
   - Calculates input power
   - Computes efficiency = P_out / P_in
   - Calculates power loss

6. **Report Generation** (if requested)
   - Creates HTML report
   - Includes power statistics table
   - Shows efficiency metrics

#### Use Cases

- **Power Supply Testing** - Characterize DC-DC converters
- **Battery Life Estimation** - Measure device power consumption
- **Efficiency Measurement** - Analyze power converter efficiency
- **Load Profiling** - Understand dynamic power consumption
- **Power Budget Analysis** - Verify system power requirements

#### References

- IEC 61000-4-7: Harmonics and interharmonics measurement
- IEEE 1459-2010: Definitions for measurement of electric power

## Signal Integrity Audit Workflow

### signal_integrity_audit()

Comprehensive signal integrity analysis workflow.

#### Purpose

Performs complete signal integrity audit including:

- Eye diagram generation and analysis
- Jitter decomposition (random vs deterministic)
- Time Interval Error (TIE) measurement
- Margin analysis against standard masks
- Dominant noise source identification
- Bit error rate estimation

Use this workflow when debugging high-speed digital signals, validating SerDes performance, or ensuring signal quality for compliance.

#### Function Signature

```python
osc.signal_integrity_audit(
    trace,                          # Data signal to analyze
    clock_trace=None,               # Optional clock signal
    *,
    bit_rate=None,                  # Bit rate in bits/second
    mask=None,                      # Eye mask standard
    report=None                     # Optional HTML report
) -> dict[str, Any]
```

#### Parameters

- **trace** (`WaveformTrace`) - Data signal to analyze. High-speed digital signal.
- **clock_trace** (`WaveformTrace`, optional) - Recovered clock or reference clock. If `None`, clock is recovered from data.
- **bit_rate** (`float`, optional) - Bit rate in bits/second. If `None`, auto-detected.
- **mask** (`str`, optional) - Optional eye mask standard for margin testing: `'PCIe'`, `'USB'`, `'SATA'`, `'Ethernet'`, etc.
- **report** (`str`, optional) - Path to save HTML report.

#### Returns

`dict[str, Any]` - Dictionary containing:

- **eye_height** (`float`) - Eye opening height in volts
- **eye_width** (`float`) - Eye opening width in seconds
- **jitter_rms** (`float`) - RMS jitter in seconds
- **jitter_pp** (`float`) - Peak-to-peak jitter in seconds
- **tie** (`ndarray`) - Time Interval Error array
- **tie_rms** (`float`) - RMS of TIE in seconds
- **margin_to_mask** (`float` or `None`) - Margin to specified mask (if provided)
- **dominant_jitter_source** (`str`) - `'random'` or `'deterministic'`
- **bit_error_rate_estimate** (`float`) - Estimated BER from eye closure
- **snr_db** (`float`) - Signal-to-noise ratio in dB
- **bit_rate** (`float`) - Detected or specified bit rate
- **unit_interval** (`float`) - Unit interval (1/bit_rate) in seconds

#### Examples

##### Basic Signal Integrity Audit

```python
import oscura as osc

# Load high-speed data signal
trace = osc.load("high_speed_data.wfm")

# Perform signal integrity audit
result = osc.signal_integrity_audit(trace, bit_rate=1e9)  # 1 Gbps

print(f"Bit Rate:          {result['bit_rate']/1e9:.2f} Gbps")
print(f"Eye Height:        {result['eye_height']*1e3:.1f} mV")
print(f"Eye Width:         {result['eye_width']*1e12:.1f} ps")
print(f"RMS Jitter:        {result['jitter_rms']*1e12:.2f} ps")
print(f"Peak-Peak Jitter:  {result['jitter_pp']*1e12:.2f} ps")
print(f"Dominant Jitter:   {result['dominant_jitter_source']}")
print(f"SNR:               {result['snr_db']:.1f} dB")
print(f"Estimated BER:     {result['bit_error_rate_estimate']:.2e}")
```

##### PCIe Signal Quality Check

```python
import oscura as osc

# Load PCIe lane capture
trace = osc.load("pcie_lane0.wfm")

# Audit with PCIe mask
result = osc.signal_integrity_audit(
    trace,
    bit_rate=5e9,  # PCIe Gen2: 5 Gbps
    mask='PCIe',
    report='pcie_si_report.html'
)

print(f"PCIe Gen2 Signal Integrity:")
print(f"  Eye Height:     {result['eye_height']*1e3:.1f} mV")
print(f"  Eye Width:      {result['eye_width']*1e12:.1f} ps")
print(f"  Margin to Mask: {result['margin_to_mask']*1e3:.1f} mV")
print(f"  RMS Jitter:     {result['jitter_rms']*1e12:.2f} ps")

# Check against PCIe specs
if result['jitter_rms']*1e12 < 10:  # PCIe typically requires <10 ps RMS
    print("✓ Jitter within PCIe specification")
else:
    print("✗ Jitter exceeds PCIe specification")
```

##### Jitter Analysis

```python
import oscura as osc
import matplotlib.pyplot as plt

# Load signal
trace = osc.load("data_signal.wfm")

# Analyze signal integrity
result = osc.signal_integrity_audit(trace, bit_rate=2.5e9)

# Plot TIE histogram
tie_ps = result['tie'] * 1e12  # Convert to picoseconds

plt.figure(figsize=(10, 6))
plt.hist(tie_ps, bins=50, density=True, alpha=0.7)
plt.xlabel('Time Interval Error (ps)')
plt.ylabel('Probability Density')
plt.title(f"TIE Histogram - {result['dominant_jitter_source'].title()} Jitter Dominant")
plt.axvline(result['jitter_rms']*1e12, color='r',
            linestyle='--', label=f'RMS: {result["jitter_rms"]*1e12:.2f} ps')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Analyze jitter sources
print(f"\nJitter Analysis:")
print(f"  RMS Jitter:     {result['jitter_rms']*1e12:.2f} ps")
print(f"  P-P Jitter:     {result['jitter_pp']*1e12:.2f} ps")
print(f"  Jitter Ratio:   {result['jitter_pp']/result['jitter_rms']:.2f}")
print(f"  Dominant Type:  {result['dominant_jitter_source']}")

if result['dominant_jitter_source'] == 'random':
    print("\n  Random jitter dominant - likely thermal or shot noise")
    print("  Consider: reducing noise, improving power supply decoupling")
else:
    print("\n  Deterministic jitter dominant - likely ISI or crosstalk")
    print("  Consider: improving signal routing, adding equalization")
```

##### Multi-Lane Comparison

```python
import oscura as osc

# Test multiple PCIe lanes
lanes = ['lane0', 'lane1', 'lane2', 'lane3']
results = {}

print("PCIe Multi-Lane Signal Integrity:")
print(f"{'Lane':<8} {'Eye (mV)':<12} {'Jitter (ps)':<14} {'SNR (dB)':<10} {'BER'}")

for lane in lanes:
    trace = osc.load(f"pcie_{lane}.wfm")
    result = osc.signal_integrity_audit(trace, bit_rate=5e9)
    results[lane] = result

    print(f"{lane:<8} "
          f"{result['eye_height']*1e3:<12.1f} "
          f"{result['jitter_rms']*1e12:<14.2f} "
          f"{result['snr_db']:<10.1f} "
          f"{result['bit_error_rate_estimate']:.2e}")

# Find worst lane
worst_lane = min(results.items(), key=lambda x: x[1]['eye_height'])
print(f"\nWorst lane: {worst_lane[0]} "
      f"(eye height: {worst_lane[1]['eye_height']*1e3:.1f} mV)")
```

##### Clock and Data Analysis

```python
import oscura as osc

# Load clock and data separately
clock = osc.load("recovered_clock.wfm")
data = osc.load("data_signal.wfm")

# Analyze with explicit clock
result = osc.signal_integrity_audit(
    data,
    clock_trace=clock,
    bit_rate=10e9  # 10 Gbps
)

print(f"Clock/Data Analysis:")
print(f"  Eye Height:     {result['eye_height']*1e3:.1f} mV")
print(f"  Eye Width:      {result['eye_width']*1e12:.1f} ps")
print(f"  Unit Interval:  {result['unit_interval']*1e12:.1f} ps")
print(f"  Eye Width %:    {result['eye_width']/result['unit_interval']*100:.1f}%")

# Eye width should be >60% of UI for good signal integrity
if result['eye_width']/result['unit_interval'] > 0.6:
    print("✓ Adequate eye width")
else:
    print("✗ Eye width below 60% of UI")
```

#### Analysis Steps Performed

The `signal_integrity_audit()` workflow performs these analysis steps:

1. **Clock Recovery** (if not provided)
   - Detects dominant frequency in data
   - Recovers bit rate and clock
   - Estimates unit interval

2. **Eye Diagram Generation**
   - Overlays multiple bit periods
   - Calculates eye opening dimensions
   - Identifies eye height and width

3. **Jitter Measurement**
   - Measures edge-to-edge timing variations
   - Calculates RMS and peak-to-peak jitter
   - Computes Time Interval Error (TIE)

4. **Jitter Decomposition**
   - Separates random from deterministic jitter
   - Analyzes jitter histogram shape
   - Identifies dominant jitter source

5. **Noise Analysis**
   - Calculates signal-to-noise ratio
   - Estimates voltage noise levels
   - Computes eye height reduction

6. **Bit Error Rate Estimation**
   - Uses eye closure to estimate BER
   - Applies Gaussian approximation
   - Provides quality metric

7. **Mask Testing** (if mask specified)
   - Loads standard eye mask
   - Calculates margin to mask
   - Identifies mask violations

#### Use Cases

- **High-Speed SerDes Testing** - PCIe, USB, SATA, Ethernet
- **Signal Quality Verification** - Validate link performance
- **Jitter Analysis** - Identify and characterize jitter sources
- **BER Prediction** - Estimate bit error rates
- **Compliance Testing** - Test against industry standards

#### Supported Eye Masks

- **PCIe** - PCI Express (Gen1-Gen5)
- **USB** - USB 2.0, 3.0, 3.1
- **SATA** - Serial ATA
- **Ethernet** - 1000BASE-T, 10GBASE-T
- **HDMI** - HDMI/DVI

#### References

- JEDEC Standard No. 65B Section 4.3: Eye diagrams
- TIA-568.2-D: Signal integrity for high-speed data
- IEEE 1596.3-1996: Low-Voltage Differential Signals

## Workflow Combinations

### Combining Multiple Workflows

Workflows can be chained together for comprehensive analysis.

#### Power Supply Complete Validation

```python
import oscura as osc

# Load measurements
v_in = osc.load("psu_input_v.wfm")
i_in = osc.load("psu_input_i.wfm")
v_out = osc.load("psu_output_v.wfm")
i_out = osc.load("psu_output_i.wfm")
emissions = osc.load("psu_emissions.wfm")

# 1. Power analysis workflow
power_result = osc.power_analysis(
    v_out, i_out,
    input_voltage=v_in,
    input_current=i_in
)

print("Power Analysis:")
print(f"  Efficiency: {power_result['efficiency']:.1f}%")
print(f"  Output Power: {power_result['output_power_avg']*1e3:.2f} mW")

# 2. EMC compliance workflow
emc_result = osc.emc_compliance_test(
    emissions,
    standard='FCC_Part15_ClassB'
)

print(f"\nEMC Compliance:")
print(f"  Status: {emc_result['status']}")
print(f"  Margin: {emc_result['margin_to_limit']:.1f} dB")

# 3. Signal integrity on output
si_result = osc.signal_integrity_audit(v_out)

print(f"\nOutput Signal Quality:")
print(f"  SNR: {si_result['snr_db']:.1f} dB")

# Overall assessment
if (power_result['efficiency'] > 85 and
    emc_result['status'] == 'PASS' and
    si_result['snr_db'] > 40):
    print("\n✓ Power supply passes all requirements")
else:
    print("\n✗ Power supply requires improvement")
```

#### Digital Interface Complete Test

```python
import oscura as osc

# Load interface signals
tx_output = osc.load("transmitter_output.wfm")
rx_input = osc.load("receiver_input.wfm")
protocol_data = osc.load("protocol_capture.wfm")

# 1. Buffer characterization on transmitter
tx_result = osc.characterize_buffer(tx_output)
print(f"Transmitter:")
print(f"  Logic Family: {tx_result['logic_family']}")
print(f"  Rise Time: {tx_result['rise_time']*1e9:.2f} ns")
print(f"  Status: {tx_result['status']}")

# 2. Protocol debugging
proto_result = osc.debug_protocol(protocol_data)
print(f"\nProtocol:")
print(f"  Type: {proto_result['protocol']}")
print(f"  Error Rate: {proto_result['statistics']['error_rate']*100:.2f}%")

# 3. Signal integrity on received signal
si_result = osc.signal_integrity_audit(rx_input, bit_rate=1e6)
print(f"\nReceiver Signal Integrity:")
print(f"  Eye Height: {si_result['eye_height']*1e3:.1f} mV")
print(f"  Jitter: {si_result['jitter_rms']*1e12:.2f} ps")

# Combined assessment
if (tx_result['status'] == 'PASS' and
    proto_result['statistics']['error_rate'] < 0.01 and
    si_result['eye_height'] > 0.3):
    print("\n✓ Interface meets requirements")
```

## Best Practices

### Sample Rate Requirements

```python
import oscura as osc

# Ensure adequate sample rate for each workflow
trace = osc.load("signal.wfm")
sample_rate = trace.metadata.sample_rate

# Buffer characterization: >10x edge bandwidth
edge_bandwidth = 0.35 / rise_time  # Estimate from rise time
min_rate_buffer = edge_bandwidth * 10
print(f"Buffer characterization requires: {min_rate_buffer/1e6:.1f} MS/s")

# Protocol debugging: >4x baud rate (Nyquist + margin)
baud_rate = 115200
min_rate_protocol = baud_rate * 4
print(f"Protocol debugging requires: {min_rate_protocol/1e3:.1f} kS/s")

# Signal integrity: >20x bit rate for accurate eye diagrams
bit_rate = 1e9
min_rate_si = bit_rate * 20
print(f"Signal integrity audit requires: {min_rate_si/1e6:.1f} MS/s")
```

### Error Handling

```python
import oscura as osc

try:
    trace = osc.load("measurement.wfm")
    result = osc.characterize_buffer(trace)
except osc.AnalysisError as e:
    print(f"Analysis failed: {e}")
    # Handle specific error conditions
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Batch Processing

```python
import oscura as osc
from pathlib import Path

# Process all files in directory
measurement_dir = Path("measurements")
results = {}

for wfm_file in measurement_dir.glob("*.wfm"):
    try:
        trace = osc.load(wfm_file)
        result = osc.characterize_buffer(trace)
        results[wfm_file.stem] = result
        print(f"✓ {wfm_file.name}: {result['status']}")
    except Exception as e:
        print(f"✗ {wfm_file.name}: {e}")
```

## API Reference Summary

### Workflow Functions

- `osc.characterize_buffer(trace, *, reference_trace=None, logic_family=None, thresholds=None, report=None)` - Digital buffer characterization
- `osc.debug_protocol(trace, *, protocol=None, context_samples=100, error_types=None, decode_all=False)` - Protocol debugging with auto-detect
- `osc.emc_compliance_test(trace, *, standard='FCC_Part15_ClassB', frequency_range=None, detector='peak', report=None)` - EMC compliance testing
- `osc.power_analysis(voltage, current, *, input_voltage=None, input_current=None, report=None)` - Power consumption analysis
- `osc.signal_integrity_audit(trace, clock_trace=None, *, bit_rate=None, mask=None, report=None)` - Signal integrity audit

## See Also

- [Analysis API](analysis.md) - Individual analysis functions
- [Power Analysis API](power-analysis.md) - Detailed power analysis functions
- [Visualization API](visualization.md) - Plotting and visualization
- [Export API](export.md) - Exporting results
- [Loader API](loader.md) - Loading waveform files
