# Common Workflows

This guide demonstrates complete workflows for typical hardware reverse engineering tasks using Oscura.

## Workflow 1: Unknown Protocol Reverse Engineering

**Scenario:** You've captured traffic from an IoT device but don't know the protocol structure.

### Step 1: Collect Captures in Different States

```python
from oscura.sessions import BlackBoxSession

# Create RE session with hypothesis tracking
session = BlackBoxSession(
    name="Smart Thermostat RE",
    description="Reverse engineering proprietary thermostat protocol"
)

# Add captures from different device states
session.add_recording("power_on", "captures/boot_sequence.bin")
session.add_recording("idle_20C", "captures/idle_20_degrees.bin")
session.add_recording("heating_on", "captures/heating_active.bin")
session.add_recording("cooling_on", "captures/cooling_active.bin")
session.add_recording("setpoint_change", "captures/temp_adjustment.bin")
session.add_recording("schedule_update", "captures/schedule_change.bin")
```

### Step 2: Differential Analysis

```python
# Compare idle vs heating states
diff_heating = session.compare("idle_20C", "heating_on")

print(f"Changed fields: {len(diff_heating.changed_fields)}")
print(f"Static header: {diff_heating.static_prefix.hex()}")

# Identify which bytes change between states
for field in diff_heating.changed_fields:
    print(f"Byte {field.offset}: {field.idle_value:02x} -> {field.heating_value:02x}")

# Compare multiple state transitions
transitions = session.compare_multiple([
    ("idle_20C", "heating_on"),
    ("idle_20C", "cooling_on"),
    ("idle_20C", "setpoint_change")
])

# Find fields that change in specific patterns
session.identify_control_fields(transitions)
```

### Step 3: Protocol Structure Inference

```python
# Automatic message structure detection
spec = session.generate_protocol_spec()

# Review inferred structure
print(f"Message format:")
print(f"  Header: {spec.header_bytes} bytes - {spec.header_pattern.hex()}")
print(f"  Total length: {spec.message_length} bytes")
print(f"  Fields: {len(spec.fields)}")

for field in spec.fields:
    print(f"    {field.name}: offset={field.offset}, "
          f"size={field.size}, type={field.inferred_type}")

# CRC/checksum auto-recovery
if spec.checksum_field:
    print(f"\nChecksum detected at offset {spec.checksum_field.offset}")
    print(f"  Algorithm: {spec.checksum_field.algorithm}")
    print(f"  Polynomial: 0x{spec.checksum_field.polynomial:X}")
    print(f"  Initial value: 0x{spec.checksum_field.init:X}")
```

### Step 4: Hypothesis Testing

```python
# Test hypotheses about field meanings
session.add_hypothesis(
    field_name="byte_5",
    hypothesis="Operating mode (0=idle, 1=heating, 2=cooling)",
    evidence={"heating_on": 0x01, "cooling_on": 0x02, "idle_20C": 0x00}
)

# Validate hypothesis against all captures
validation = session.validate_hypothesis("byte_5")
print(f"Hypothesis confidence: {validation.confidence:.2%}")

# Export hypothesis audit trail
session.export_hypotheses("analysis_trail.json")
```

### Step 5: Export Artifacts

```python
# Generate Wireshark dissector
session.export_results("dissector", "thermostat_proto.lua")

# Generate Scapy layer for packet crafting
session.export_results("scapy", "thermostat_layer.py")

# Generate comprehensive report
session.export_results("report", "re_report.html")

# Export protocol specification
session.export_results("spec", "PROTOCOL_SPEC.md")
```

---

## Workflow 2: CAN Bus Reverse Engineering

**Scenario:** Analyzing vehicle CAN bus to understand message definitions.

### Step 1: Load CAN Capture

```python
from oscura.sessions import CANSession

# Create CAN analysis session
session = CANSession(
    bitrate=500000,
    fd_mode=False,
    name="Vehicle CAN Analysis"
)

# Load capture (supports BLF, ASC, PCAP, etc.)
session.load("captures/vehicle_drive.blf")

# Basic statistics
stats = session.get_statistics()
print(f"Total messages: {stats.message_count}")
print(f"Unique IDs: {len(stats.unique_ids)}")
print(f"Duration: {stats.duration:.2f} seconds")
print(f"Bus load: {stats.bus_load:.1%}")
```

### Step 2: Identify Message Patterns

```python
# Automatic message classification
patterns = session.find_patterns()

# Periodic messages (constant rate)
for msg_id, period in patterns.periodic_messages.items():
    print(f"ID 0x{msg_id:03X}: periodic at {period}ms")

# Event-driven messages
for msg_id, triggers in patterns.event_driven_messages.items():
    print(f"ID 0x{msg_id:03X}: triggered by {triggers}")

# State-dependent messages
for msg_id, states in patterns.state_dependent_messages.items():
    print(f"ID 0x{msg_id:03X}: present in states {states}")
```

### Step 3: Signal Extraction

```python
# Automatic signal boundary detection
signals = session.extract_signals()

for signal in signals:
    print(f"\nSignal: {signal.name}")
    print(f"  CAN ID: 0x{signal.can_id:03X}")
    print(f"  Start bit: {signal.start_bit}")
    print(f"  Length: {signal.length} bits")
    print(f"  Byte order: {signal.byte_order}")
    print(f"  Value type: {signal.value_type}")
    print(f"  Range: {signal.min_value} to {signal.max_value}")
    print(f"  Resolution: {signal.resolution}")
    print(f"  Unit: {signal.unit}")
```

### Step 4: Correlate with Vehicle Behavior

```python
# Correlate signals with external events
session.add_event_marker(12.5, "brake_applied")
session.add_event_marker(15.2, "accelerator_pressed")
session.add_event_marker(20.1, "turn_signal_left")

# Find signals that change with events
correlations = session.correlate_with_events()

for signal, events in correlations.items():
    print(f"{signal.name} correlates with: {events}")

# Identify counter/sequence signals
counters = session.find_counters()
checksums = session.find_checksums()
```

### Step 5: Generate DBC File

```python
# Export DBC file (no manual signal definition needed!)
session.export_dbc(
    output="vehicle_protocol.dbc",
    include_comments=True,
    validation=True
)

# Export to other formats
session.export_kcd("vehicle_protocol.kcd")  # CANdb++
session.export_arxml("vehicle_protocol.arxml")  # AUTOSAR
```

---

## Workflow 3: Signal Integrity Validation

**Scenario:** Verify signal quality meets specification requirements.

### Complete Signal Quality Check

```python
from oscura.workflows import signal_integrity_analysis

# Load high-speed signal
waveform = load("usb_hs_capture.wfm")

# Comprehensive signal quality analysis
results = signal_integrity_analysis(
    waveform=waveform,
    specification="USB 2.0 High-Speed",
    checks=[
        "rise_time",
        "fall_time",
        "overshoot",
        "undershoot",
        "ringing",
        "jitter",
        "eye_diagram",
        "crossing_voltage",
        "differential_pairs"
    ]
)

# Check compliance
print(f"\nSignal Quality Report:")
print(f"  Overall: {'PASS' if results.passes else 'FAIL'}")
print(f"  Score: {results.quality_score:.1f}/100")

# Detailed metrics
print(f"\nTiming Metrics:")
print(f"  Rise time: {results.rise_time:.2f} ns (spec: <{results.spec.rise_time_max} ns)")
print(f"  Fall time: {results.fall_time:.2f} ns (spec: <{results.spec.fall_time_max} ns)")

print(f"\nVoltage Metrics:")
print(f"  Overshoot: {results.overshoot:.1f}% (spec: <{results.spec.overshoot_max}%)")
print(f"  Undershoot: {results.undershoot:.1f}% (spec: <{results.spec.undershoot_max}%)")

print(f"\nJitter Analysis:")
print(f"  RMS jitter: {results.jitter_rms:.2f} ps")
print(f"  Peak-to-peak: {results.jitter_pp:.2f} ps")
print(f"  TIE: {results.tie_max:.2f} ps")

# Eye diagram quality
print(f"\nEye Diagram:")
print(f"  Eye height: {results.eye_height:.3f} V (spec: >{results.spec.eye_height_min} V)")
print(f"  Eye width: {results.eye_width:.2f} UI (spec: >{results.spec.eye_width_min} UI)")
print(f"  Mask margin: {results.mask_margin:.1f}%")

# Generate detailed report with plots
results.export_report("signal_quality_report.html")
```

---

## Workflow 4: Side-Channel / Power Analysis

**Scenario:** Recover cryptographic keys from power consumption traces.

### Correlation Power Analysis (CPA)

```python
from oscura.workflows import power_analysis
import numpy as np

# Load power traces and plaintexts
traces = np.load("power_traces.npy")  # Shape: (num_traces, samples_per_trace)
plaintexts = np.load("plaintexts.npy")  # Shape: (num_traces, 16)

# Perform CPA attack on AES-128
results = power_analysis(
    traces=traces,
    plaintexts=plaintexts,
    algorithm="AES-128",
    attack_type="CPA",
    target_operation="SubBytes",
    target_byte=0,  # Attack first key byte
    leakage_model="HammingWeight"
)

# Display key recovery results
print(f"\nKey Recovery Results:")
print(f"  Recovered key byte: 0x{results.key_byte:02x}")
print(f"  Confidence: {results.confidence:.2%}")
print(f"  Correlation: {results.max_correlation:.4f}")
print(f"  Traces required: {results.traces_needed}")
print(f"  Analysis time: {results.analysis_time:.2f}s")

# Visualize attack
results.plot_correlation_traces()
results.plot_key_rank_evolution()

# Attack all 16 key bytes
full_key = power_analysis.attack_full_key(
    traces=traces,
    plaintexts=plaintexts,
    algorithm="AES-128",
    parallel=True
)

print(f"\nFull recovered key: {full_key.hex()}")
```

### Template Attack

```python
from oscura.workflows import template_attack

# Profiling phase (known key)
profiling_traces = np.load("profiling_traces.npy")
profiling_keys = np.load("profiling_keys.npy")

templates = template_attack.create_templates(
    traces=profiling_traces,
    keys=profiling_keys,
    algorithm="AES-128",
    points_of_interest=500  # Select 500 most informative samples
)

# Attack phase (unknown key)
attack_traces = np.load("attack_traces.npy")
attack_results = template_attack.attack(
    traces=attack_traces,
    templates=templates,
    plaintexts=attack_plaintexts
)

print(f"Recovered key: {attack_results.key.hex()}")
print(f"Success probability: {attack_results.probability:.2%}")
```

---

## Workflow 5: Batch Processing Multiple Captures

**Scenario:** Process hundreds of captures with the same analysis.

### Parallel Batch Analysis

```python
# NOTE: Use workflows or manual iteration in v0.6
# from oscura.workflows import batch_analyze, BatchConfig

# Configure batch processing
config = BatchConfig(
    input_dir="./captures/field_test/",
    output_dir="./results/",
    pattern="*.wfm",
    parallel=True,
    num_workers=8,
    fail_fast=False  # Continue on errors
)

# Define analysis function
def analyze_uart(waveform):
    """Analyze UART traffic from waveform."""
    from oscura.analyzers.protocols import UARTDecoder

    decoder = UARTDecoder(baud_rate=115200)
    messages = decoder.decode(waveform)

    return {
        "message_count": len(messages),
        "unique_patterns": len(set(m.data for m in messages)),
        "error_rate": sum(1 for m in messages if m.has_error) / len(messages),
        "avg_message_length": np.mean([len(m.data) for m in messages])
    }

# Run batch analysis
results = batch_analyze(
    config=config,
    analysis_func=analyze_uart,
    progress_bar=True
)

# Aggregate results
summary = results.aggregate(
    metrics=["message_count", "error_rate", "unique_patterns"],
    aggregations=["mean", "std", "min", "max"]
)

print(f"\nBatch Analysis Summary ({len(results)} captures):")
print(f"  Total messages: {summary.message_count.sum()}")
print(f"  Average error rate: {summary.error_rate.mean:.2%}")
print(f"  Unique patterns: {summary.unique_patterns.sum()}")

# Export results
results.to_csv("batch_analysis_results.csv")
results.to_html("batch_analysis_report.html")

# Find outliers
outliers = results.find_outliers(
    metric="error_rate",
    threshold=3.0  # 3 standard deviations
)

print(f"\nOutliers detected: {len(outliers)}")
for outlier in outliers:
    print(f"  {outlier.filename}: error_rate={outlier.error_rate:.2%}")
```

---

## Workflow 6: Protocol Fuzzing

**Scenario:** Test device robustness with malformed inputs.

### Grammar-Based Fuzzing

```python
from oscura.testing import ProtocolFuzzer

# Create fuzzer from protocol specification
fuzzer = ProtocolFuzzer.from_spec(
    spec_file="thermostat_proto.yaml",
    transport="serial",
    target_device="/dev/ttyUSB0"
)

# Configure fuzzing strategy
fuzzer.configure(
    strategy="grammar_aware",  # Respect protocol grammar
    mutations=[
        "bit_flip",
        "boundary_values",
        "checksum_corruption",
        "length_manipulation",
        "sequence_reorder"
    ],
    iterations=10000,
    crash_detection=True,
    timeout_per_test=1.0  # seconds
)

# Run fuzzing campaign
results = fuzzer.run()

# Analyze results
print(f"\nFuzzing Results:")
print(f"  Test cases: {results.test_count}")
print(f"  Crashes: {results.crash_count}")
print(f"  Hangs: {results.hang_count}")
print(f"  Unique failures: {results.unique_failures}")
print(f"  Code coverage: {results.coverage:.1%}")

# Export crash reproducers
for crash in results.crashes:
    crash.save_reproducer(f"crashes/crash_{crash.id}.bin")

# Generate fuzzing report
results.export_report("fuzzing_report.html")
```

---

## Workflow 7: Multi-Protocol Session Correlation

**Scenario:** Device uses multiple protocols simultaneously (UART + SPI + CAN).

### Cross-Protocol Analysis

```python
from oscura.sessions import MultiProtocolSession

# Create multi-protocol session
session = MultiProtocolSession(name="ECU Complete Analysis")

# Add different protocol captures with time alignment
session.add_protocol(
    name="diagnostic_uart",
    protocol_type="UART",
    capture_file="uart_debug.wfm",
    config={"baud_rate": 115200}
)

session.add_protocol(
    name="sensor_spi",
    protocol_type="SPI",
    capture_file="spi_sensors.vcd",
    config={"mode": 0, "bit_order": "MSB"}
)

session.add_protocol(
    name="vehicle_can",
    protocol_type="CAN",
    capture_file="can_bus.blf",
    config={"bitrate": 500000}
)

# Time-align all protocols (find common timestamp reference)
session.align_protocols(reference="diagnostic_uart")

# Find cross-protocol correlations
correlations = session.find_correlations()

for correlation in correlations:
    print(f"\nCorrelation found:")
    print(f"  Protocol 1: {correlation.protocol1} - {correlation.message1}")
    print(f"  Protocol 2: {correlation.protocol2} - {correlation.message2}")
    print(f"  Time delta: {correlation.time_delta:.6f} s")
    print(f"  Confidence: {correlation.confidence:.2%}")

# Generate unified timeline visualization
session.export_timeline("timeline.html")

# Export comprehensive multi-protocol report
session.export_report("multi_protocol_analysis.html")
```

---

## Next Steps

- **[Tutorials](../tutorials/)** - Step-by-step guides for specific protocols
- **[API Reference](../api/)** - Detailed API documentation
- **[Protocol Catalog](../protocols/)** - Supported protocols and examples
- **[FAQ](../faq/)** - Frequently asked questions

## Best Practices

### Capture Quality

1. **Use appropriate sample rates:** At least 10x the signal frequency
2. **Capture sufficient data:** Include complete transactions, not fragments
3. **Document test conditions:** Record device state, temperature, power supply
4. **Multiple captures:** Collect data in different states for differential analysis

### Analysis Strategy

1. **Start simple:** Use known-good captures to validate setup
2. **Incremental complexity:** Decode known protocols first, then unknown
3. **Hypothesis-driven:** Form hypotheses, test them, document results
4. **Cross-validation:** Verify findings with multiple captures
5. **Reproducibility:** Save all analysis scripts and parameters

### Performance Optimization

1. **Use streaming for large files:** Avoid loading entire capture into memory
2. **Enable parallel processing:** Utilize multi-core CPUs for batch work
3. **Cache intermediate results:** Save decoded messages for reuse
4. **Profile bottlenecks:** Use `oscura.profiling` to identify slow operations

### Security Considerations

1. **Validate external inputs:** Never trust captured data implicitly
2. **Sandbox analysis:** Run unknown protocol analysis in isolated environment
3. **Protect sensitive data:** Encrypt/redact proprietary protocol details
4. **Responsible disclosure:** Follow ethical guidelines for vulnerability reporting
