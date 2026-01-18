# UDP Packet Analysis Demos

Demonstrates **comprehensive network packet reverse engineering** and protocol analysis capabilities for UDP-based protocols.

---

## Files in This Demo

1. **`comprehensive_udp_analysis.py`** - **COMPREHENSIVE**
   - Complete UDP packet analysis and protocol reverse engineering
   - Protocol dissection (Ethernet/IP/UDP layers)
   - Traffic metrics (throughput, jitter, latency, loss)
   - Payload analysis (entropy, patterns, clustering)
   - Field inference (delimiters, types, checksums, sequences)
   - Protocol fingerprinting
   - Differential analysis
   - Multi-format export (JSON, CSV, HDF5, MATLAB, Markdown, HTML)
   - Wireshark dissector generation
   - **Use this for unknown protocol reverse engineering**

2. **`generate_test_packets.py`**
   - Generate synthetic UDP packet captures for testing
   - Configurable packet structure and patterns
   - Creates valid PCAP files
   - **Use this for testing without real captures**

---

## Quick Start

### 1. Generate Test Packets

```bash
# Generate test PCAP file with synthetic traffic
uv run python demos/06_udp_packet_analysis/generate_test_packets.py \
    --output test_packets.pcap \
    --count 1000 \
    --packet-size 1024

# Generate with custom parameters
uv run python demos/06_udp_packet_analysis/generate_test_packets.py \
    --output custom.pcap \
    --count 5000 \
    --packet-size 512 \
    --src-port 5000 \
    --dst-port 6000
```

**Output**: Creates PCAP file with synthetic UDP packets

### 2. Comprehensive Analysis

```bash
# Analyze UDP packets from PCAP file
uv run python demos/06_udp_packet_analysis/comprehensive_udp_analysis.py \
    --pcap-file test_packets.pcap \
    --output-dir udp_analysis \
    --verbose

# Analyze with differential mode (compare before/after changes)
uv run python demos/06_udp_packet_analysis/comprehensive_udp_analysis.py \
    --pcap-file capture.pcap \
    --output-dir analysis \
    --differential
```

**Output**: Creates in `udp_analysis/`:

- `summary.json` - Complete analysis results
- `traffic_metrics.csv` - Throughput, jitter, latency
- `payload_analysis.json` - Entropy, patterns, clustering
- `field_inference.json` - Inferred protocol structure
- `protocol_fingerprint.json` - Protocol identification
- `packets.h5` - HDF5 export of packet data
- `packets.mat` - MATLAB export
- `report.md` - Markdown analysis report
- `report.html` - HTML report with visualizations
- `dissector.lua` - Wireshark dissector (if structure inferred)

---

## What This Demo Shows

### Comprehensive UDP Packet Analysis

**Categories analyzed**:

1. **Protocol Dissection**
   - Ethernet frame parsing
   - IP header extraction
   - UDP header parsing
   - Payload extraction
   - Layer validation

2. **Traffic Metrics**
   - Packet rate (packets/sec)
   - Throughput (bits/sec, bytes/sec)
   - Inter-packet jitter
   - Latency measurements
   - Packet loss detection
   - Burst analysis

3. **Payload Analysis**
   - Entropy calculation
   - Pattern detection
   - Data clustering
   - Compression ratio
   - Randomness testing
   - Repeated sequences

4. **Field Inference**
   - Delimiter detection
   - Field type inference (int, float, string, binary)
   - Checksum/CRC detection
   - Sequence number detection
   - Timestamp detection
   - Length field detection

5. **Protocol Fingerprinting**
   - Protocol identification
   - Version detection
   - Vendor fingerprinting
   - Known protocol matching
   - Behavior analysis

6. **Differential Analysis**
   - Before/after comparison
   - Field change detection
   - Protocol evolution tracking
   - Anomaly detection

7. **Export Formats**
   - JSON (analysis results)
   - CSV (time-series metrics)
   - HDF5 (binary packet data)
   - MATLAB (.mat files)
   - Markdown reports
   - HTML reports with plots
   - Wireshark dissector (.lua)

8. **Wireshark Integration**
   - Automatic dissector generation
   - Field definitions
   - Protocol tree structure
   - Color rules
   - Expert info annotations

### Protocol Reverse Engineering

**Capabilities demonstrated**:

- **Unknown protocol analysis**: Infer structure from packet captures
- **Pattern recognition**: Detect repeated headers, footers, markers
- **Field boundary detection**: Find where fields start/end
- **Type inference**: Determine data types (integers, floats, strings)
- **Checksum detection**: Identify and validate checksums/CRCs
- **Sequence tracking**: Detect sequence numbers and ordering
- **Timing correlation**: Find time-based patterns
- **State machine learning**: Infer protocol state transitions

---

## Understanding the Outputs

### Analysis Summary (summary.json)

```json
{
  "file_info": {
    "filename": "test_packets.pcap",
    "total_packets": 1000,
    "file_size": 1024000,
    "duration": 10.0
  },
  "traffic_metrics": {
    "packet_rate": 100.0,
    "throughput_bps": 819200,
    "throughput_Mbps": 0.819,
    "avg_jitter_ms": 2.5,
    "max_jitter_ms": 15.3,
    "packet_loss_pct": 0.0
  },
  "payload_analysis": {
    "avg_entropy": 7.2,
    "compression_ratio": 0.85,
    "unique_patterns": 47,
    "repeated_sequences": 12
  },
  "field_inference": {
    "likely_header_size": 16,
    "detected_fields": [
      {
        "offset": 0,
        "size": 4,
        "type": "uint32",
        "name": "sequence_number"
      },
      {
        "offset": 4,
        "size": 4,
        "type": "uint32",
        "name": "timestamp"
      }
    ]
  }
}
```

### Wireshark Dissector (dissector.lua)

```lua
-- Auto-generated Wireshark dissector for unknown protocol
custom_proto = Proto("custom", "Custom Protocol")

-- Field definitions
local f_sequence = ProtoField.uint32("custom.sequence", "Sequence Number")
local f_timestamp = ProtoField.uint32("custom.timestamp", "Timestamp")

custom_proto.fields = {f_sequence, f_timestamp}

function custom_proto.dissector(buffer, pinfo, tree)
    pinfo.cols.protocol = "CUSTOM"
    local subtree = tree:add(custom_proto, buffer())

    subtree:add(f_sequence, buffer(0, 4))
    subtree:add(f_timestamp, buffer(4, 4))
end

-- Register dissector
local udp_table = DissectorTable.get("udp.port")
udp_table:add(5000, custom_proto)
```

### Traffic Metrics Report

```
UDP Traffic Analysis Report
================================================

CAPTURE INFO
------------------------------------------------
File: test_packets.pcap
Duration: 10.0 seconds
Total packets: 1000
File size: 1.0 MB

TRAFFIC METRICS
------------------------------------------------
Packet rate: 100.0 packets/sec
Throughput: 0.82 Mbps (819.2 Kbps)
Average jitter: 2.5 ms
Maximum jitter: 15.3 ms
Packet loss: 0.0%

PAYLOAD ANALYSIS
------------------------------------------------
Average entropy: 7.2 bits/byte (90% of max)
Compression ratio: 0.85
Unique patterns: 47
Repeated sequences: 12

INFERRED PROTOCOL STRUCTURE
------------------------------------------------
Header size: 16 bytes
Fields detected: 2

Field 1: Sequence Number
  Offset: 0
  Size: 4 bytes
  Type: uint32
  Range: 0-999

Field 2: Timestamp
  Offset: 4
  Size: 4 bytes
  Type: uint32
  Range: 0-10000
```

---

## Use Cases

### 1. Unknown Protocol Reverse Engineering

**Scenario**: You have PCAP captures of an unknown proprietary protocol.

```bash
# Run comprehensive analysis
uv run python demos/06_udp_packet_analysis/comprehensive_udp_analysis.py \
    --pcap-file unknown_protocol.pcap \
    --output-dir re_analysis \
    --verbose

# Review results
cat re_analysis/field_inference.json  # See inferred structure
cat re_analysis/protocol_fingerprint.json  # See protocol ID

# Use generated Wireshark dissector
cp re_analysis/dissector.lua ~/.config/wireshark/plugins/
# Restart Wireshark, protocol now decoded!
```

### 2. IoT Device Communication Analysis

**Scenario**: Analyzing smart home device communication.

```bash
# Capture traffic with tcpdump/Wireshark first
sudo tcpdump -i eth0 -w iot_capture.pcap udp

# Analyze the capture
uv run python demos/06_udp_packet_analysis/comprehensive_udp_analysis.py \
    --pcap-file iot_capture.pcap \
    --output-dir iot_analysis
```

### 3. Network Protocol Debugging

**Scenario**: Your custom protocol isn't working correctly.

```bash
# Analyze working capture
uv run python demos/06_udp_packet_analysis/comprehensive_udp_analysis.py \
    --pcap-file working.pcap \
    --output-dir working_analysis

# Analyze broken capture
uv run python demos/06_udp_packet_analysis/comprehensive_udp_analysis.py \
    --pcap-file broken.pcap \
    --output-dir broken_analysis

# Use differential mode to compare
uv run python demos/06_udp_packet_analysis/comprehensive_udp_analysis.py \
    --pcap-file broken.pcap \
    --output-dir diff_analysis \
    --differential \
    --reference working.pcap
```

### 4. Performance Testing

**Scenario**: Validating network performance.

```bash
# Generate test traffic
uv run python demos/06_udp_packet_analysis/generate_test_packets.py \
    --output test.pcap \
    --count 10000 \
    --packet-size 1024

# Analyze performance
uv run python demos/06_udp_packet_analysis/comprehensive_udp_analysis.py \
    --pcap-file test.pcap \
    --output-dir perf_analysis

# Check traffic_metrics.csv for throughput, jitter, loss
```

---

## Advanced Features

### Field Inference Algorithm

The analyzer uses multiple techniques to infer protocol structure:

1. **Boundary detection**: Find field boundaries by looking for constant/changing regions
2. **Type inference**: Analyze value distributions to infer types (int, float, string)
3. **Checksum detection**: Test common CRC/checksum algorithms
4. **Sequence detection**: Look for monotonically increasing values
5. **Length field detection**: Find fields that encode payload length
6. **Delimiter detection**: Identify repeated patterns that separate fields

### Protocol Fingerprinting

Identifies known protocols by:

- Port numbers
- Packet size distributions
- Header patterns
- Timing characteristics
- Payload entropy
- Behavioral patterns

### Differential Analysis

Compares two captures to find:

- New/removed fields
- Changed field values
- Protocol version differences
- Behavioral changes
- Performance regressions

---

## Performance Notes

### Analysis Speed

- **Small captures** (<1000 packets): <1 second
- **Medium captures** (1K-100K packets): 1-10 seconds
- **Large captures** (>100K packets): 10-60 seconds

### Memory Usage

- Packet data kept in memory during analysis
- ~1 KB per packet typical
- 100K packets = ~100 MB RAM

### Optimization Tips

- Use filtered captures (only UDP traffic)
- Split large captures into chunks
- Use HDF5 export for large datasets

---

## Common Issues

### Issue: "No packets found in PCAP"

**Solution**: Ensure PCAP contains UDP packets, check with Wireshark first

### Issue: "Field inference failed"

**Solution**: Need more packets (>100 recommended) with varied data

### Issue: "Wireshark dissector doesn't work"

**Solution**:

- Check dissector syntax
- Ensure correct port number
- Reload Wireshark plugins (Ctrl+Shift+L)

### Issue: "Memory error with large PCAP"

**Solution**: Split PCAP into smaller chunks using `editcap`:

```bash
editcap -c 10000 large.pcap small.pcap
```

---

## Related Documentation

- **Main demos**: `demos/README.md`
- **Signal RE demos**: `demos/17_signal_reverse_engineering/`
- **Protocol inference**: `demos/07_protocol_inference/`

---

## Extending the Demos

### Add Custom Protocol Fingerprint

```python
# In comprehensive_udp_analysis.py
KNOWN_PROTOCOLS = {
    "my_protocol": {
        "port": 12345,
        "header_pattern": b"\x00\x01\x02\x03",
        "min_size": 64,
        "max_size": 1024,
    }
}
```

### Add Custom Field Inference

```python
def detect_custom_field(packets):
    """Detect custom protocol field."""
    # Your detection logic
    if pattern_matches:
        return {
            "offset": offset,
            "size": size,
            "type": "custom_type",
        }
```

### Add Custom Export Format

```python
def export_custom_format(analysis_results, output_path):
    """Export to custom format."""
    # Your export logic
    with open(output_path, 'w') as f:
        # Write custom format
        pass
```

---

## Integration with Wireshark

### Using Generated Dissectors

1. **Generate dissector**:

   ```bash
   uv run python demos/06_udp_packet_analysis/comprehensive_udp_analysis.py \
       --pcap-file capture.pcap \
       --output-dir analysis
   ```

2. **Install dissector**:

   ```bash
   # Linux/Mac
   cp analysis/dissector.lua ~/.config/wireshark/plugins/

   # Windows
   copy analysis\dissector.lua %APPDATA%\Wireshark\plugins\
   ```

3. **Reload Wireshark**:
   - Open Wireshark
   - Press `Ctrl+Shift+L` to reload Lua plugins
   - Open your PCAP file
   - Protocol now automatically decoded!

### Enhancing Generated Dissectors

The generated dissector is a starting point. You can enhance it by:

- Adding expert info annotations
- Implementing field validation
- Adding subtree structures
- Creating color rules
- Adding context menus

---

**Last Updated**: 2026-01-16
**Status**: Production-ready, fully validated
