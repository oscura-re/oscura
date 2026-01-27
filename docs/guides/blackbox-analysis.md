# Black-Box Protocol Analysis Guide

**Version**: 0.6.0
**Last Updated**: 2026-01-25

Complete guide to reverse engineering unknown protocols using Oscura's BlackBoxSession.

---

## Overview

Black-box protocol analysis is the process of reverse engineering unknown communication protocols through differential analysis, pattern recognition, and machine learning techniques.

**Use Cases**:

- IoT device protocol reverse engineering
- Proprietary protocol understanding
- Security vulnerability discovery
- Right-to-repair device replication
- Commercial intelligence

**Key Capabilities**:

- Differential analysis (byte-level comparison)
- Field hypothesis generation
- State machine inference
- CRC/checksum reverse engineering
- Protocol specification generation
- Wireshark dissector export

---

## Quick Start

### Basic Protocol Analysis

```python
from oscura.sessions import BlackBoxSession
# NOTE: Direct loading recommended in v0.6
import oscura as osc

# Create analysis session
session = BlackBoxSession(name="IoT Device Protocol Analysis")

# Add recordings from different stimuli
session.add_recording("baseline", osc.load("idle.bin"))
session.add_recording("button_press", osc.load("button.bin"))
session.add_recording("temp_25C", osc.load("temp25.bin"))
session.add_recording("temp_30C", osc.load("temp30.bin"))

# Compare recordings to find differences
diff = session.compare("baseline", "button_press")
print(f"Changed bytes: {diff.changed_bytes}")
print(f"Similarity: {diff.similarity_score:.2%}")

# Analyze changed regions
for start, end, description in diff.changed_regions:
    print(f"Bytes {start}-{end}: {description}")
```

### Generate Protocol Specification

```python
# Infer protocol structure
spec = session.generate_protocol_spec()

print(f"Protocol: {spec['name']}")
print(f"Message size: {spec['message_size']} bytes")
print(f"Fields detected: {len(spec['fields'])}")

# Examine field hypotheses
for field in spec['fields']:
    print(f"{field.name}: offset={field.offset}, "
          f"length={field.length}, type={field.field_type}, "
          f"confidence={field.confidence:.2f}")
```

### Export Results

```python
# Export as Markdown report
session.export_results("report", "analysis_report.md")

# Export as Wireshark dissector
session.export_results("dissector", "protocol.lua")

# Export as JSON specification
session.export_results("json", "protocol_spec.json")

# Export raw data as CSV
session.export_results("csv", "comparison_data.csv")
```

---

## Core Concepts

### 1. Differential Analysis

Compare recordings to identify protocol fields that change based on stimuli.

**Principle**: By comparing messages captured under different conditions, you can infer which bytes encode which information.

**Example**:

```python
# Record device at different temperatures
session.add_recording("temp_20C", osc.load("temp20.bin"))
session.add_recording("temp_25C", osc.load("temp25.bin"))
session.add_recording("temp_30C", osc.load("temp30.bin"))

# Compare to find temperature field
diff_20_25 = session.compare("temp_20C", "temp_25C")
diff_25_30 = session.compare("temp_25C", "temp_30C")

# Bytes that changed in both comparisons likely encode temperature
```

### 2. Field Hypothesis Generation

Automatically generate hypotheses about field types and meanings.

**Field Types**:

- **Counter**: Monotonically increasing values
- **Constant**: Fixed values (headers, magic numbers)
- **Checksum**: Computed from other fields
- **Data**: Variable payload data
- **Unknown**: Unclassified fields

**Example**:

```python
spec = session.generate_protocol_spec()

for field in spec['fields']:
    if field.field_type == "counter":
        print(f"Counter field at offset {field.offset}")
        print(f"Evidence: {field.evidence}")
    elif field.field_type == "checksum":
        print(f"Checksum at offset {field.offset}, algorithm: {field.evidence['algorithm']}")
```

### 3. State Machine Inference

Learn protocol state machines from message sequences.

**Principle**: Protocols have states (IDLE, SYNC, DATA, etc.) and transitions triggered by events.

**Example**:

```python
# Infer state machine from recordings
sm = session.infer_state_machine()

print(f"States: {len(sm.states)}")
print(f"Transitions: {len(sm.transitions)}")
print(f"Initial state: {sm.initial_state}")

# Visualize state machine
for state in sm.states:
    print(f"State {state.name}: {state.description}")
    for transition in state.outgoing:
        print(f"  -> {transition.target} on {transition.trigger}")
```

### 4. CRC/Checksum Detection

Automatically detect and reverse engineer checksums.

**Supported Algorithms**:

- CRC-8 (various polynomials)
- CRC-16 (CCITT, MODBUS, etc.)
- CRC-32 (IEEE, etc.)
- Simple checksums (XOR, SUM)

**Example**:

```python
spec = session.generate_protocol_spec()

# Find checksum fields
for field in spec['fields']:
    if field.field_type == "checksum":
        print(f"Checksum: {field.evidence['algorithm']}")
        print(f"Position: offset {field.offset}, length {field.length}")
        print(f"Covers: bytes {field.evidence['start']}-{field.evidence['end']}")
```

---

## Complete Workflow

### Step 1: Data Collection

Capture protocol data under controlled conditions.

```python
from oscura.sessions import BlackBoxSession
# NOTE: Direct loading recommended in v0.6
import oscura as osc

session = BlackBoxSession(name="Smart Lock Protocol")

# Baseline: device idle
session.add_recording("idle", osc.load("captures/idle.bin"))

# Stimulus 1: unlock command
session.add_recording("unlock", osc.load("captures/unlock.bin"))

# Stimulus 2: lock command
session.add_recording("lock", osc.load("captures/lock.bin"))

# Stimulus 3: invalid PIN
session.add_recording("invalid_pin", osc.load("captures/invalid_pin.bin"))

# Stimulus 4: valid PIN
session.add_recording("valid_pin", osc.load("captures/valid_pin.bin"))
```

**Best Practices**:

- Capture multiple samples per stimulus (at least 5-10)
- Use controlled test environment
- Document stimulus conditions
- Include baseline (idle) recording
- Capture edge cases (errors, boundaries)

---

### Step 2: Differential Analysis

Compare recordings systematically.

```python
# Compare all recordings to baseline
baseline_comparisons = {}
for name in session.list_recordings():
    if name != "idle":
        diff = session.compare("idle", name)
        baseline_comparisons[name] = diff
        print(f"{name}: {diff.changed_bytes} bytes changed, "
              f"similarity={diff.similarity_score:.2%}")

# Find consistently changing regions
print("\nChanged Regions:")
for name, diff in baseline_comparisons.items():
    print(f"\n{name}:")
    for start, end, desc in diff.changed_regions:
        print(f"  Bytes {start:3d}-{end:3d}: {desc}")
```

---

### Step 3: Field Hypothesis Generation

Generate and evaluate field hypotheses.

```python
# Generate protocol specification
spec = session.generate_protocol_spec()

print(f"\nProtocol Specification: {spec['name']}")
print(f"Message Size: {spec['message_size']} bytes")
print(f"Fields Detected: {len(spec['fields'])}\n")

# Examine high-confidence fields
high_confidence = [f for f in spec['fields'] if f.confidence > 0.8]
print(f"High-confidence fields ({len(high_confidence)}):")
for field in high_confidence:
    print(f"  {field.name}:")
    print(f"    Offset: {field.offset}")
    print(f"    Length: {field.length} bytes")
    print(f"    Type: {field.field_type}")
    print(f"    Confidence: {field.confidence:.2%}")
    print(f"    Evidence: {field.evidence}\n")

# Examine uncertain fields
uncertain = [f for f in spec['fields'] if f.confidence < 0.5]
print(f"\nUncertain fields ({len(uncertain)}) - need more data:")
for field in uncertain:
    print(f"  Offset {field.offset}: {field.field_type} (confidence={field.confidence:.2%})")
```

---

### Step 4: State Machine Inference

Learn protocol behavior.

```python
# Infer state machine
sm = session.infer_state_machine()

print(f"\nState Machine:")
print(f"  States: {len(sm.states)}")
print(f"  Transitions: {len(sm.transitions)}")
print(f"  Initial State: {sm.initial_state}\n")

# Examine states
print("States:")
for state in sm.states:
    print(f"  {state.name}:")
    print(f"    Description: {state.description}")
    print(f"    Average duration: {state.duration_avg:.4f}s")
    print(f"    Frequency: {state.frequency:.2%}")

# Examine transitions
print("\nTransitions:")
for transition in sm.transitions:
    print(f"  {transition.source} -> {transition.target}")
    print(f"    Trigger: {transition.trigger}")
    print(f"    Probability: {transition.probability:.2%}")
```

---

### Step 5: CRC/Checksum Verification

Validate and document checksums.

```python
from oscura.inference.crc_reverse import verify_crc

# Check for common CRC algorithms
crc_results = session.detect_checksums()

print("\nChecksum Detection Results:")
for result in crc_results:
    print(f"  Algorithm: {result['algorithm']}")
    print(f"  Position: offset {result['offset']}, length {result['length']}")
    print(f"  Coverage: bytes {result['covers_start']}-{result['covers_end']}")
    print(f"  Confidence: {result['confidence']:.2%}")

    if result['algorithm'].startswith('CRC'):
        print(f"  Polynomial: 0x{result['polynomial']:X}")
        print(f"  Initial value: 0x{result['init']:X}")
        print(f"  Final XOR: 0x{result['xorout']:X}")
```

---

### Step 6: Generate Documentation

Export comprehensive protocol documentation.

```python
# Markdown report
session.export_results("report", "smart_lock_protocol.md")

# JSON specification
session.export_results("json", "smart_lock_protocol.json")

# Wireshark dissector
session.export_results("dissector", "smart_lock.lua")

# CSV data export
session.export_results("csv", "field_comparisons.csv")

print("\nExported:")
print("  - smart_lock_protocol.md (human-readable report)")
print("  - smart_lock_protocol.json (machine-readable spec)")
print("  - smart_lock.lua (Wireshark dissector)")
print("  - field_comparisons.csv (raw comparison data)")
```

---

## Advanced Techniques

### Multi-Stimulus Correlation

Correlate changes across multiple stimuli to improve confidence.

```python
# Test multiple related conditions
session.add_recording("temp_10C", osc.load("temp10.bin"))
session.add_recording("temp_20C", osc.load("temp20.bin"))
session.add_recording("temp_30C", osc.load("temp30.bin"))
session.add_recording("temp_40C", osc.load("temp40.bin"))

# Analyze correlation
temp_diffs = [
    session.compare("temp_10C", "temp_20C"),
    session.compare("temp_20C", "temp_30C"),
    session.compare("temp_30C", "temp_40C"),
]

# Find bytes that change consistently
consistent_changes = {}
for i, diff in enumerate(temp_diffs):
    for start, end, desc in diff.changed_regions:
        key = (start, end)
        consistent_changes[key] = consistent_changes.get(key, 0) + 1

# Regions that changed in ALL comparisons
for (start, end), count in consistent_changes.items():
    if count == len(temp_diffs):
        print(f"Consistent change: bytes {start}-{end} (temperature field candidate)")
```

---

### Message Boundary Detection

Identify message framing and boundaries.

```python
from oscura.inference.message_format import infer_format

# Load binary data
with osc.load("capture.bin") as source:
    trace = source.read()

# Infer message format
format_info = infer_format(trace.data)

print(f"Detected message format:")
print(f"  Start pattern: 0x{format_info.start_pattern:X}")
print(f"  Message length: {format_info.message_length} bytes")
print(f"  Length field: offset {format_info.length_offset}, size {format_info.length_size}")
print(f"  Header size: {format_info.header_size} bytes")
print(f"  Footer size: {format_info.footer_size} bytes")
```

---

### Protocol DSL Generation

Generate protocol definition in DSL format.

```python
# Export as Protocol DSL
session.export_results("dsl", "protocol.yaml")

# Example DSL output:
"""
protocol:
  name: "Smart Lock Protocol"
  version: "1.0"

  messages:
    - name: UNLOCK
      id: 0x01
      fields:
        - name: sequence
          type: uint16
          offset: 0
        - name: command
          type: uint8
          offset: 2
          value: 0x01
        - name: pin
          type: bytes
          offset: 3
          length: 4
        - name: checksum
          type: crc16
          offset: 7
          algorithm: CRC-16/CCITT-FALSE
"""
```

---

### State Machine Validation

Validate inferred state machine against new captures.

```python
# Infer state machine from training data
sm = session.infer_state_machine()

# Load new test data
test_session = BlackBoxSession(name="Validation")
test_session.add_recording("test1", osc.load("test1.bin"))

# Validate state machine predictions
validation_results = sm.validate(test_session.get_recording("test1"))

print(f"Validation accuracy: {validation_results.accuracy:.2%}")
print(f"Prediction errors: {validation_results.errors}")
print(f"Unexpected transitions: {validation_results.unexpected_transitions}")
```

---

## Wireshark Integration

### Generate Dissector

```python
# Export Wireshark dissector
session.export_results("dissector", "myprotocol.lua")

# The generated dissector includes:
# - Protocol name and description
# - Field definitions with types
# - Checksum validation
# - Subtree organization
```

### Install Dissector

```bash
# Copy to Wireshark plugins directory
mkdir -p ~/.local/lib/wireshark/plugins/
cp myprotocol.lua ~/.local/lib/wireshark/plugins/

# Or for system-wide installation
sudo cp myprotocol.lua /usr/lib/x86_64-linux-gnu/wireshark/plugins/

# Restart Wireshark to load plugin
```

### Use Dissector

1. Open Wireshark
2. Load capture file (PCAP/PCAPNG)
3. Find your protocol in protocol hierarchy
4. View decoded fields in packet details pane
5. Filter by protocol: `myprotocol`

---

## Troubleshooting

### Issue: "No field hypotheses generated"

**Causes**:

- Insufficient data (need multiple recordings)
- All recordings identical (no differential signal)
- Data too noisy or random

**Solutions**:

```python
# 1. Add more recordings with varied stimuli
session.add_recording("stimulus_1", osc.load("stim1.bin"))
session.add_recording("stimulus_2", osc.load("stim2.bin"))
session.add_recording("stimulus_3", osc.load("stim3.bin"))

# 2. Check data quality
for name in session.list_recordings():
    trace = session.get_recording(name)
    print(f"{name}: {len(trace.data)} bytes")

# 3. Verify recordings differ
diff = session.compare("stimulus_1", "stimulus_2")
if diff.changed_bytes == 0:
    print("WARNING: Recordings identical - add different stimuli")
```

---

### Issue: "Low confidence in field types"

**Causes**:

- Ambiguous patterns
- Complex encoding
- Insufficient samples

**Solutions**:

```python
# 1. Add more recordings for same stimulus
for i in range(10):
    session.add_recording(f"unlock_{i}", FileSource(f"unlock_{i}.bin"))

# 2. Use targeted analysis
spec = session.generate_protocol_spec()
uncertain = [f for f in spec['fields'] if f.confidence < 0.5]

print(f"Uncertain fields: {len(uncertain)}")
for field in uncertain:
    print(f"  Offset {field.offset}: evidence={field.evidence}")
    # Manually verify with additional captures
```

---

### Issue: "State machine too complex"

**Causes**:

- Noisy data creating false states
- Natural protocol complexity
- Incorrect message boundaries

**Solutions**:

```python
# 1. Filter noise before inference
from oscura.utils.filtering import low_pass

filtered_recordings = {}
for name in session.list_recordings():
    trace = session.get_recording(name)
    filtered = low_pass(trace, cutoff=1e6)
    filtered_recordings[name] = filtered

# 2. Increase minimum state duration
sm = session.infer_state_machine(min_duration=0.001)  # 1ms minimum

# 3. Reduce state count
sm = session.infer_state_machine(max_states=5)
```

---

### Issue: "Checksum detection failed"

**Causes**:

- Custom CRC algorithm
- Checksum over non-contiguous fields
- Multiple checksums

**Solutions**:

```python
from oscura.inference.crc_reverse import verify_crc, find_crc_polynomial

# 1. Try custom polynomial search
trace = session.get_recording("test")
polynomial = find_crc_polynomial(trace.data, width=16)
print(f"Detected polynomial: 0x{polynomial:X}")

# 2. Manually verify checksums
def check_custom_crc(data):
    # Your custom CRC logic
    payload = data[:-2]
    expected_crc = int.from_bytes(data[-2:], 'big')
    computed_crc = custom_crc_function(payload)
    return computed_crc == expected_crc

# Test on samples
trace = session.get_recording("test")
if check_custom_crc(trace.data):
    print("Custom CRC validated")
```

---

## Best Practices

### 1. Systematic Data Collection

**Good**:

```python
# Organized, documented collection
session = BlackBoxSession(name="Device XYZ Protocol")

# Baseline
session.add_recording("baseline_idle", osc.load("baseline/idle_1.bin"))

# Single variable tests
session.add_recording("test_unlock", osc.load("tests/unlock_1.bin"))
session.add_recording("test_lock", osc.load("tests/lock_1.bin"))

# Multiple samples
for i in range(5):
    session.add_recording(f"unlock_{i}", FileSource(f"samples/unlock_{i}.bin"))
```

**Bad**:

```python
# Disorganized collection
session.add_recording("test1", osc.load("data.bin"))
session.add_recording("test2", osc.load("capture.bin"))
session.add_recording("other", osc.load("temp.bin"))
# No clear relationship between recordings
```

---

### 2. Validate Hypotheses

**Always verify field hypotheses**:

```python
spec = session.generate_protocol_spec()

# For each field, validate with new captures
for field in spec['fields']:
    print(f"\nValidating field: {field.name}")
    print(f"  Hypothesis: {field.field_type} at offset {field.offset}")

    # Load validation data
    validation = osc.load("validation.bin").read()

    # Check if hypothesis holds
    field_data = validation.data[field.offset:field.offset + field.length]
    print(f"  Observed value: 0x{field_data.hex()}")

    # Manual verification against expected behavior
```

---

### 3. Iterative Refinement

**Process is iterative**:

1. **Initial collection** → Basic patterns
2. **Targeted collection** → Refine hypotheses
3. **Validation** → Confirm findings
4. **Documentation** → Record results

```python
# Iteration 1: Broad exploration
session_v1 = BlackBoxSession(name="Exploration")
# ... add varied recordings ...
spec_v1 = session_v1.generate_protocol_spec()

# Iteration 2: Targeted investigation
session_v2 = BlackBoxSession(name="Refinement")
# ... add focused recordings based on v1 findings ...
spec_v2 = session_v2.generate_protocol_spec()

# Compare results
print(f"V1 fields: {len(spec_v1['fields'])}")
print(f"V2 fields: {len(spec_v2['fields'])}")
print(f"High-confidence in V2: {sum(1 for f in spec_v2['fields'] if f.confidence > 0.8)}")
```

---

### 4. Document Everything

**Maintain analysis journal**:

```python
# Export results at each stage
session.export_results("report", f"analysis_iteration_{i}.md")
session.export_results("json", f"spec_iteration_{i}.json")

# Include metadata
metadata = {
    "iteration": i,
    "date": "2026-01-20",
    "recordings": len(session.list_recordings()),
    "hypothesis_count": len(spec['fields']),
    "confidence_avg": sum(f.confidence for f in spec['fields']) / len(spec['fields']),
    "notes": "Focused on temperature field validation"
}

import json
with open(f"metadata_iteration_{i}.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

---

## Example: Complete IoT Device Analysis

Full end-to-end example.

```python
from oscura.sessions import BlackBoxSession
# NOTE: Direct loading recommended in v0.6
import oscura as osc
from pathlib import Path

# Setup
session = BlackBoxSession(name="Smart Thermostat Protocol RE")

# Phase 1: Data Collection
print("Phase 1: Data Collection")
capture_dir = Path("captures/thermostat")

session.add_recording("idle", FileSource(capture_dir / "idle.bin"))
session.add_recording("set_temp_20", FileSource(capture_dir / "set_temp_20.bin"))
session.add_recording("set_temp_25", FileSource(capture_dir / "set_temp_25.bin"))
session.add_recording("set_fan_low", FileSource(capture_dir / "fan_low.bin"))
session.add_recording("set_fan_high", FileSource(capture_dir / "fan_high.bin"))

print(f"Loaded {len(session.list_recordings())} recordings")

# Phase 2: Differential Analysis
print("\nPhase 2: Differential Analysis")
comparisons = {}
for name in session.list_recordings():
    if name != "idle":
        diff = session.compare("idle", name)
        comparisons[name] = diff
        print(f"  {name}: {diff.changed_bytes} bytes changed")

# Phase 3: Field Hypothesis
print("\nPhase 3: Field Hypothesis Generation")
spec = session.generate_protocol_spec()
print(f"  Detected {len(spec['fields'])} fields")
print(f"  Message size: {spec['message_size']} bytes")

# Phase 4: State Machine
print("\nPhase 4: State Machine Inference")
sm = session.infer_state_machine()
print(f"  States: {len(sm.states)}")
print(f"  Transitions: {len(sm.transitions)}")

# Phase 5: Checksum Detection
print("\nPhase 5: Checksum Detection")
checksums = session.detect_checksums()
for cs in checksums:
    print(f"  {cs['algorithm']} at offset {cs['offset']}")

# Phase 6: Export Results
print("\nPhase 6: Export Results")
output_dir = Path("analysis/thermostat")
output_dir.mkdir(parents=True, exist_ok=True)

session.export_results("report", output_dir / "protocol_analysis.md")
session.export_results("json", output_dir / "protocol_spec.json")
session.export_results("dissector", output_dir / "thermostat.lua")

print(f"  Results exported to {output_dir}")
print("\nAnalysis complete!")
```

---

## Related Documentation

- [Hardware Acquisition Guide](hardware-acquisition.md) - Capture data from devices
- [Session Management API](../api/session-management.md) - Full API reference
- [Protocol Inference Demo](../../demos/07_protocol_inference/) - Working examples
- [Migration Guide](../migration/v0-to-v1.md) - Upgrade from older versions

---

**Next Steps**:

1. Collect protocol captures from your target device
2. Follow the complete workflow
3. Validate and refine hypotheses iteratively
4. Export and use Wireshark dissector for ongoing analysis
