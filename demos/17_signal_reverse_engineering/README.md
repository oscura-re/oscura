# Signal Reverse Engineering Demos

Demonstrates **automated signal analysis and protocol inference** from unknown binary captures using machine learning and pattern recognition techniques.

---

## Files in This Demo

1. **`reverse_engineer_tool.py`** - **MAIN TOOL**
   - Automated signal reverse engineering
   - Pattern detection and extraction
   - State machine learning
   - CRC/checksum detection
   - Timing correlation
   - Message boundary inference
   - Protocol structure extraction
   - Export to JSON, CSV, HDF5, Wireshark dissector
   - **Use this for unknown signal/protocol analysis**

2. **`exploratory_analysis.py`**
   - Interactive signal exploration
   - Visual data inspection
   - Statistical analysis
   - Pattern visualization
   - Correlation analysis
   - **Use this for initial investigation**

3. **`comprehensive_re.py`** - **ULTIMATE**
   - Complete reverse engineering pipeline
   - All analysis techniques combined
   - Multi-level pattern detection
   - Advanced state machine inference
   - Comprehensive reporting
   - **Use this for thorough analysis**

---

## Quick Start

### 1. Main Reverse Engineering Tool

```bash
# Analyze unknown binary signal
uv run python demos/17_signal_reverse_engineering/reverse_engineer_tool.py \
    unknown_signal.bin \
    --output-dir re_results \
    --verbose

# With custom parameters
uv run python demos/17_signal_reverse_engineering/reverse_engineer_tool.py \
    signal.bin \
    --output-dir analysis \
    --sample-rate 100e6 \
    --threshold auto \
    --export-wireshark
```

**Output**: Creates in `re_results/`:

- `summary.json` - Complete RE results
- `patterns.json` - Detected patterns
- `state_machine.json` - Inferred state transitions
- `timing.json` - Timing analysis
- `protocol_structure.json` - Inferred protocol
- `report.md` - Markdown report
- `report.html` - HTML report with visualizations
- `dissector.lua` - Wireshark dissector (if --export-wireshark)

### 2. Exploratory Analysis

```bash
# Interactive exploration
uv run python demos/17_signal_reverse_engineering/exploratory_analysis.py \
    signal.bin \
    --interactive

# Batch analysis
uv run python demos/17_signal_reverse_engineering/exploratory_analysis.py \
    signal.bin \
    --output-dir exploration \
    --verbose
```

**Output**: Interactive plots and statistical summaries

### 3. Comprehensive Analysis

```bash
# Ultimate comprehensive RE
uv run python demos/17_signal_reverse_engineering/comprehensive_re.py \
    signal.bin \
    --output-dir ultimate_re \
    --all-techniques \
    --verbose

# With specific techniques
uv run python demos/17_signal_reverse_engineering/comprehensive_re.py \
    signal.bin \
    --output-dir analysis \
    --techniques pattern,state,crc,timing
```

**Output**: Complete analysis with all RE techniques applied

---

## What This Demo Shows

### Automated Reverse Engineering

**Core capabilities**:

1. **Pattern Detection**
   - Repeated sequences
   - Preambles and sync words
   - Header patterns
   - Footer patterns
   - Delimiter detection
   - Field boundaries

2. **State Machine Learning**
   - State identification
   - Transition detection
   - State probability estimation
   - Valid/invalid transitions
   - Protocol flow reconstruction

3. **CRC/Checksum Detection**
   - Common CRC algorithms (CRC8, CRC16, CRC32)
   - Custom polynomial detection
   - Checksum validation
   - Error detection analysis

4. **Timing Analysis**
   - Symbol timing extraction
   - Baud rate detection
   - Clock recovery
   - Timing correlation
   - Jitter measurement
   - Phase analysis

5. **Message Boundary Inference**
   - Start/stop detection
   - Length field identification
   - Delimiter-based boundaries
   - Timing-based boundaries
   - Protocol framing

6. **Protocol Structure Extraction**
   - Header structure
   - Payload organization
   - Field types and sizes
   - Nested structures
   - Protocol layering

7. **Visualization**
   - Signal plots
   - Pattern heatmaps
   - State transition diagrams
   - Timing diagrams
   - Correlation matrices

8. **Export Formats**
   - JSON (analysis results)
   - CSV (time-series data)
   - HDF5 (binary data)
   - Markdown reports
   - HTML reports with plots
   - Wireshark dissectors

### Machine Learning Techniques

**Algorithms used**:

- **Clustering**: K-means for state identification
- **Classification**: Pattern recognition
- **Sequence analysis**: Hidden Markov Models
- **Time series**: Autocorrelation, cross-correlation
- **Feature extraction**: PCA, statistical moments
- **Anomaly detection**: Isolation forests

---

## Understanding the Outputs

### Summary Report (summary.json)

```json
{
  "file_info": {
    "filename": "unknown_signal.bin",
    "file_size": 2900000000,
    "samples": 382000000,
    "duration": 3.82
  },
  "patterns_detected": {
    "count": 47,
    "preambles": [
      {
        "pattern": "0xAA55",
        "occurrences": 1234,
        "frequency": 323.0,
        "confidence": 0.95
      }
    ],
    "headers": [
      {
        "pattern": "0x0001",
        "size": 16,
        "position": "after_preamble"
      }
    ]
  },
  "state_machine": {
    "states": 5,
    "transitions": 12,
    "initial_state": "IDLE",
    "states_identified": [
      {
        "id": 0,
        "name": "IDLE",
        "duration_avg": 0.001,
        "frequency": 0.45
      },
      {
        "id": 1,
        "name": "SYNC",
        "duration_avg": 0.0001,
        "frequency": 0.25
      }
    ]
  },
  "timing_analysis": {
    "symbol_rate": 100000000,
    "clock_detected": true,
    "jitter_rms": 0.000001,
    "timing_accuracy": 0.999
  }
}
```

### State Transition Diagram

```
+------+
| IDLE |<----------------------+
+--+---+                       |
   | preamble                  |
   v                           |
+------+                       |
| SYNC |                       |
+--+---+                       |
   | header                    |
   v                           |
+----------+                   |
| DATA_RX  |                   |
+--+-------+                   |
   | checksum_ok               |
   v                           |
+----------+                   |
| VALIDATE |-------------------+
+----------+    done
```

### Protocol Structure Visualization

```
Packet Structure (64 bytes):
+------------+----------+--------------+---------+
| Preamble   | Header   | Payload      | CRC     |
| 2 bytes    | 4 bytes  | 56 bytes     | 2 bytes |
| 0xAA55     | Seq+Len  | Data         | CRC16   |
+------------+----------+--------------+---------+

Header Details:
  Bytes 0-1: Sequence number (uint16)
  Bytes 2-3: Payload length (uint16)

Payload: Variable data (application specific)

CRC: CRC-16/CCITT-FALSE over header + payload
```

---

## Use Cases

### 1. Embedded System Protocol RE

**Scenario**: Reverse engineer communication between microcontroller and peripherals.

```bash
# Capture logic analyzer data to binary file first
# Then analyze
uv run python demos/17_signal_reverse_engineering/reverse_engineer_tool.py \
    mcu_comm.bin \
    --output-dir mcu_re \
    --sample-rate 100e6 \
    --export-wireshark \
    --verbose

# Review results
cat mcu_re/protocol_structure.json
cat mcu_re/state_machine.json

# Use Wireshark dissector
cp mcu_re/dissector.lua ~/.config/wireshark/plugins/
```

### 2. Unknown Radio Protocol

**Scenario**: Captured RF baseband signal, need to decode.

```bash
# Start with exploratory analysis
uv run python demos/17_signal_reverse_engineering/exploratory_analysis.py \
    rf_capture.bin \
    --interactive

# Then comprehensive RE
uv run python demos/17_signal_reverse_engineering/comprehensive_re.py \
    rf_capture.bin \
    --output-dir rf_re \
    --all-techniques
```

### 3. Security Research

**Scenario**: Analyzing proprietary IoT device protocol.

```bash
# Comprehensive analysis
uv run python demos/17_signal_reverse_engineering/comprehensive_re.py \
    iot_capture.bin \
    --output-dir iot_re \
    --techniques pattern,state,crc,timing \
    --export-wireshark

# Generate detailed report
cat iot_re/report.html  # Open in browser for visualizations
```

### 4. Automotive CAN Bus Analysis

**Scenario**: Reverse engineering custom CAN protocol.

```bash
# Analyze CAN capture
uv run python demos/17_signal_reverse_engineering/reverse_engineer_tool.py \
    can_capture.bin \
    --output-dir can_re \
    --sample-rate 500000 \
    --threshold auto \
    --export-wireshark
```

---

## Advanced Features

### Pattern Detection Algorithm

Multi-level pattern detection:

1. **Byte-level patterns**: Common sequences (0xAA, 0x55, etc.)
2. **Word-level patterns**: 16/32-bit repeating values
3. **Structural patterns**: Headers, footers, delimiters
4. **Statistical patterns**: Entropy changes, distribution shifts
5. **Temporal patterns**: Periodic occurrences

### State Machine Inference

Automated state machine learning:

1. **Feature extraction**: Extract signal characteristics
2. **Clustering**: Group similar signal segments
3. **State identification**: Label clusters as states
4. **Transition detection**: Identify state changes
5. **Validation**: Verify state machine consistency
6. **Optimization**: Minimize states while preserving accuracy

### CRC Detection

Tests multiple algorithms:

- CRC-8 (various polynomials)
- CRC-16 (CCITT, MODBUS, etc.)
- CRC-32 (IEEE, BZIP2, etc.)
- Custom polynomials
- Simple checksums (sum, XOR)

### Timing Recovery

Advanced timing analysis:

1. **Edge detection**: Find symbol boundaries
2. **Histogram analysis**: Identify symbol periods
3. **PLL simulation**: Recover clock
4. **Jitter measurement**: Quantify timing variations
5. **Eye diagram**: Visualize signal quality

---

## Performance Notes

### Analysis Speed

- **Small files** (<10 MB): <10 seconds
- **Medium files** (10-100 MB): 10-60 seconds
- **Large files** (100 MB - 1 GB): 1-10 minutes
- **Very large files** (>1 GB): 10-60 minutes

### Memory Usage

- **Pattern detection**: O(n) memory
- **State machine**: O(states x samples)
- **CRC detection**: O(1) per algorithm
- **Typical**: 500 MB for 1 GB file

### Optimization Tips

- Use exploratory analysis first (faster)
- Focus on specific techniques (--techniques)
- Reduce sample rate if appropriate
- Process chunks for very large files

---

## Common Issues

### Issue: "No patterns detected"

**Solution**:

- Check data is not pure random noise
- Adjust threshold parameter
- Try different sample rates
- Verify file format

### Issue: "State machine too complex"

**Solution**:

- Reduce noise (filter data first)
- Adjust clustering parameters
- Focus on specific signal sections
- Increase minimum state duration

### Issue: "CRC detection failed"

**Solution**:

- Ensure sufficient sample data (>100 messages)
- Verify packet boundaries are correct
- Try different CRC positions
- Check for custom CRC algorithms

### Issue: "Memory error"

**Solution**:

- Process file in chunks
- Reduce clustering resolution
- Disable memory-intensive techniques
- Use 64-bit Python on large files

---

## Integration with Other Tools

### Wireshark

```bash
# Generate dissector
uv run python demos/17_signal_reverse_engineering/reverse_engineer_tool.py \
    signal.bin \
    --output-dir re \
    --export-wireshark

# Install dissector
cp re/dissector.lua ~/.config/wireshark/plugins/

# Use with UDP packet captures
# The dissector will decode your custom protocol!
```

### GNU Radio

Export timing and modulation info for GNU Radio flowgraphs:

```python
import json
with open('re/timing.json') as f:
    timing = json.load(f)

# Use symbol_rate in GNU Radio blocks
symbol_rate = timing['symbol_rate']
```

### MATLAB/Octave

Export data for custom analysis:

```bash
# Export to MATLAB format
uv run python demos/17_signal_reverse_engineering/reverse_engineer_tool.py \
    signal.bin \
    --output-dir re \
    --export-matlab

# Load in MATLAB
load('re/analysis.mat')
```

---

## Related Documentation

- **Main demos**: `demos/README.md`
- **UDP packet analysis**: `demos/06_udp_packet_analysis/`
- **Custom DAQ formats**: `demos/03_custom_daq/`
- **Protocol inference**: `demos/07_protocol_inference/`

---

## Extending the Demos

### Add Custom Pattern Detector

```python
class CustomPatternDetector:
    """Detect custom patterns."""

    def detect(self, data: np.ndarray) -> list[dict]:
        """Detect patterns in data."""
        patterns = []

        # Your detection logic here
        for i in range(len(data) - pattern_len):
            if self.matches_pattern(data[i:i+pattern_len]):
                patterns.append({
                    "offset": i,
                    "pattern": data[i:i+pattern_len],
                    "confidence": self.calculate_confidence()
                })

        return patterns
```

### Add Custom State Machine Algorithm

```python
class CustomStateMachine:
    """Custom state machine inference."""

    def learn(self, data: np.ndarray) -> dict:
        """Learn state machine from data."""
        # Your ML algorithm here
        states = self.identify_states(data)
        transitions = self.find_transitions(data, states)

        return {
            "states": states,
            "transitions": transitions,
            "confidence": self.calculate_confidence()
        }
```

### Add Custom CRC Algorithm

```python
def detect_custom_crc(data: np.ndarray, packet_size: int) -> dict | None:
    """Detect custom CRC algorithm."""
    # Split into packets
    packets = data.reshape(-1, packet_size)

    # Test your CRC algorithm
    for packet in packets:
        payload = packet[:-2]
        crc = packet[-2:]

        if custom_crc(payload) == crc:
            return {
                "algorithm": "custom",
                "position": -2,
                "size": 2,
                "confidence": 0.95
            }

    return None
```

---

## Research Applications

### Academic Research

- Protocol reverse engineering research
- Security vulnerability analysis
- Wireless protocol forensics
- Embedded system analysis

### Industry Applications

- Automotive security (CAN bus analysis)
- IoT device security testing
- Industrial control system RE
- Legacy system documentation
- Competitive intelligence

### Security Research

- Vulnerability discovery
- Fuzzing target preparation
- Protocol implementation verification
- Malware communication analysis

---

## Best Practices

### 1. Start with Exploration

Always begin with exploratory analysis to understand the signal:

```bash
# First: explore
uv run python demos/17_signal_reverse_engineering/exploratory_analysis.py \
    signal.bin --interactive

# Then: targeted RE
uv run python demos/17_signal_reverse_engineering/reverse_engineer_tool.py \
    signal.bin --techniques pattern,timing
```

### 2. Validate Results

Always validate inferred protocol:

- Compare with known protocol documentation
- Test with multiple captures
- Verify checksums/CRCs
- Validate state transitions

### 3. Document Findings

Use generated reports:

- Markdown reports for documentation
- HTML reports for presentations
- JSON exports for automation
- Wireshark dissectors for ongoing analysis

### 4. Iterative Refinement

RE is iterative:

1. Initial analysis -> hypotheses
2. Targeted analysis -> validation
3. Manual refinement -> accuracy
4. Final validation -> deployment

---

**Last Updated**: 2026-01-16
**Status**: Production-ready, research-grade tools
