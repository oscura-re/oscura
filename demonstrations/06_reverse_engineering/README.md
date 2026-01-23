# Reverse Engineering Demonstrations

Complete demonstration suite for Oscura's reverse engineering capabilities. These demonstrations showcase the full workflow for analyzing unknown protocols, from initial capture through to exportable specifications.

## Overview

| Demo                        | Description                           | Key Features                                             |
| --------------------------- | ------------------------------------- | -------------------------------------------------------- |
| **01_unknown_protocol.py**  | Complete unknown protocol RE workflow | Differential analysis, field hypothesis, state inference |
| **02_crc_recovery.py**      | CRC polynomial recovery               | CRC-8/16/32, polynomial finding, parameter recovery      |
| **03_state_machines.py**    | State machine learning                | L\* algorithm, PTA construction, DOT export              |
| **04_field_inference.py**   | Field boundary detection              | Entropy analysis, type classification, correlation       |
| **05_pattern_discovery.py** | Message pattern recognition           | Request-response, bursts, timing analysis                |
| **06_wireshark_export.py**  | Wireshark dissector generation        | Lua dissectors, JSON/Markdown export, DBC files          |

## Quick Start

Each demonstration is self-contained and can be run independently:

```bash
# Run individual demonstration
python demonstrations/06_reverse_engineering/01_unknown_protocol.py

# Run all demonstrations in sequence
for demo in demonstrations/06_reverse_engineering/0*.py; do
    echo "Running $demo..."
    python "$demo"
done
```

## Demonstration Details

### 01_unknown_protocol.py

**Complete Unknown Protocol Reverse Engineering Workflow**

Demonstrates the full reverse engineering process:

- Generate proprietary protocol messages
- Collect and analyze message samples
- Detect field boundaries using entropy analysis
- Classify field types (constant, counter, variable)
- Identify special fields (magic, checksum, length)
- Infer protocol specification

**Capabilities Demonstrated:**

- `oscura.inference.protocol.ProtocolInferrer`
- `oscura.inference.message_format.MessageFormatInferrer`
- `oscura.inference.binary.detect_field_boundaries()`
- `oscura.inference.binary.classify_field_type()`
- `oscura.inference.sequences.find_repeated_patterns()`
- `oscura.inference.alignment.align_local()`

### 02_crc_recovery.py

**CRC Polynomial Recovery from Message Samples**

Shows how to recover CRC parameters without prior knowledge:

- Generate messages with known CRC algorithms
- Apply XOR differential technique
- Detect CRC width (8, 16, 32 bits)
- Find polynomial through pattern matching
- Recover init, XOR_out, and reflection flags
- Identify standard algorithms (CRC-16-CCITT, MODBUS, etc.)

**Capabilities Demonstrated:**

- `oscura.inference.crc_reverse.CRCReverser`
- `oscura.inference.crc_reverse.verify_crc()`
- `oscura.inference.crc_reverse.STANDARD_CRCS`

### 03_state_machines.py

**State Machine Learning Using L\_ Algorithm**

Demonstrates protocol state machine extraction:

- Generate valid and invalid message sequences
- Build Prefix Tree Acceptor (PTA)
- Apply RPNI algorithm for state merging
- Validate against positive/negative samples
- Test novel sequences
- Export to DOT format for visualization

**Capabilities Demonstrated:**

- `oscura.inference.state_machine.StateMachineInferrer`
- `oscura.inference.state_machine.RPNI`
- `oscura.inference.state_machine.PrefixTreeAcceptor`
- `oscura.inference.state_machine.accepts()`
- `oscura.inference.state_machine.to_dot()`

### 04_field_inference.py

**Automatic Field Boundary and Type Detection**

Shows field structure inference techniques:

- Generate structured messages with known layout
- Detect field boundaries via entropy transitions
- Classify field types statistically
- Identify counter fields (sequence numbers)
- Detect length fields and validate correlation
- Find checksum fields
- Validate special field detection

**Capabilities Demonstrated:**

- `oscura.inference.message_format.MessageFormatInferrer`
- `oscura.inference.binary.detect_field_boundaries()`
- `oscura.inference.binary.classify_field_type()`
- `oscura.inference.binary.detect_length_field()`
- `oscura.inference.binary.detect_checksum_field()`

### 05_pattern_discovery.py

**Message Pattern Recognition and Analysis**

Demonstrates behavioral pattern discovery:

- Generate message sequences with patterns
- Find repeated message patterns
- Detect request-response pairs
- Correlate messages into sessions
- Analyze timing patterns (periodic, bursts)
- Identify error-retry patterns

**Capabilities Demonstrated:**

- `oscura.inference.sequences.find_repeated_patterns()`
- `oscura.inference.sequences.detect_request_response()`
- `oscura.inference.sequences.correlate_sessions()`
- `oscura.inference.sequences.analyze_timing()`

### 06_wireshark_export.py

**Wireshark Dissector Generation from Inferred Protocols**

Shows protocol export and integration:

- Create protocol definitions using DSL
- Generate Wireshark Lua dissectors
- Export protocol specifications to JSON
- Generate Markdown documentation
- Analyze generated dissector code
- Provide installation instructions

**Capabilities Demonstrated:**

- `oscura.export.wireshark.WiresharkDissectorGenerator`
- `oscura.inference.protocol_dsl.ProtocolDefinition`
- `oscura.inference.protocol_dsl.FieldDefinition`
- `oscura.export.wireshark.generate_to_string()`
- `oscura.export.dbc.generate_dbc()`

## Architecture

All demonstrations follow the `BaseDemo` template pattern:

```python
class MyDemo(BaseDemo):
    def generate_test_data(self) -> dict[str, Any]:
        # Generate or load synthetic test data
        return {"data": ...}

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        # Execute demonstration with phases
        results = {}
        self.section("Phase 1: ...")
        # ... demonstration logic ...
        return results

    def validate(self, results: dict[str, Any]) -> bool:
        # Validate results
        return True
```

## Output

Each demonstration produces:

- Console output with formatted sections and results
- Validation of expected outcomes
- Generated files in `demonstrations/data/outputs/<demo_name>/`
- Return code 0 for success, 1 for failure

## Testing

The demonstrations serve as:

1. **User examples** - Show how to use reverse engineering features
2. **Integration tests** - Validate complete workflows
3. **Living documentation** - Demonstrate best practices

## Requirements

- Python 3.12+
- numpy
- Oscura reverse engineering modules:
  - `oscura.inference.protocol`
  - `oscura.inference.message_format`
  - `oscura.inference.crc_reverse`
  - `oscura.inference.state_machine`
  - `oscura.inference.sequences`
  - `oscura.export.wireshark`

## Related Documentation

- Main documentation: `docs/`
- Test suite guide: `docs/testing/test-suite-guide.md`
- Contributing guide: `CONTRIBUTING.md`
- Changelog: `CHANGELOG.md`

## Statistics

- **Total demonstrations**: 6
- **Total lines of code**: ~2,400
- **Capabilities demonstrated**: 20+
- **Coverage**: Complete reverse engineering workflow

## P0 CRITICAL Status

All demonstrations in this directory are **P0 CRITICAL** features, representing core reverse engineering capabilities essential for:

- Unknown protocol analysis
- Security research
- Right-to-repair efforts
- Defense analysis
- Commercial intelligence
